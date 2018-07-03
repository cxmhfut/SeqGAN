import tensorflow as tf


def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k,i] * input_[i]) + Bias[k]
    Args:
        :param input_: a tensor or a list of 2D, batch x n, Tensors.
        :param output_size: int, second dimension of W[i]
        :param scope: VariableScope for the created sub graph; default to "Linear".
    Returns:
        :return:
        A 2D Tensor with shape [batch x, output_size] equal to
        sum_i(input[i] * W[i]), where W[i]s are newly created matrices.
    Raise:
        ValueError: if some of the arguments are unspecified or wrong shape.
    """
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))

    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or 'SimpleLinear'):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """
    Highway Network (http://arxiv.org/abs/1505.00387)
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is non-linearity, t is transform gate, and (1 - t) is carry gate
    """
    output = None
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))
            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)
            output = t * g + (1. - t) * input_
            input_ = output

    return output


class Discriminator(object):
    """
    A CNN for text classification.
    Uses am embedding layer, followed by a convolutional, max-pooling and softmax layer
    """

    def __init__(self, config):
        # placeholder for input, output and dropout
        self.sequence_length = config.sequence_length
        self.num_classes = config.num_classes
        self.vocab_size = config.vocab_size
        self.filter_sizes = config.dis_filter_sizes
        self.num_filters = config.dis_num_filters
        self.learning_rate = config.dis_learining_rate
        self.embedding_size = config.dis_embedding_size
        self.l2_reg_lambda = config.dis_l2_reg_lambda

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        # Keeping track of 12 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

        self.W = None
        self.embedded_chars = None
        self.embedded_chars_expanded = None
        self.h_pool = None
        self.h_pool_flat = None
        self.h_highway = None
        self.h_drop = None
        self.scores = None
        self.ypred_for_auc = None
        self.predictions = None
        self.loss = None
        self.params = None
        self.train_op = None

    def build_discriminator(self):
        with tf.variable_scope('discriminator'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope('embedding'):
                self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     name='W')
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + max-pool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(self.filter_sizes, self.num_filters):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_size, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name='b')
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv'
                    )

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool'
                    )
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filter_total = sum(self.num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])

            # Add highway
            with tf.name_scope('highway'):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope('dropout'):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope('output'):
                W = tf.Variable(tf.truncated_normal([num_filter_total, self.num_classes], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')
                self.l2_loss += tf.nn.l2_loss(W)
                self.l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name='predictions')

            # CalculateMean cross-entropy loss
            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)