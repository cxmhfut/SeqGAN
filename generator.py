import tensorflow as tf


class Generator(object):
    """
        SeqGAN implementation
    """

    def __init__(self, config):
        """
        Basic set up
        Args:
              num_emb: output vocabulary size
              batch_size: batch_size for generator
              emb_dim: LSTM hidden unit dimension
              sequence_length: maximum length of input sequence
              start_token: special token index for start of sequence
              initializer: initializer for LSTM kernel and output matrix
        """
        self.num_emb = config.num_emb
        self.batch_size = config.gen_batch_size
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.sequence_length = config.sequence_length
        self.start_token = tf.constant(config.start_token, dtype=tf.int32, shape=[self.batch_size])
        self.initializer = tf.random_normal_initializer(stddev=0.1)

        self.input_seqs_pre = None
        self.input_seqs_mask = None
        self.input_seqs_adv = None
        self.rewards = None
        self.pretrain_loss = None
        self.pretrain_loss_sum = None
        self.softmax_list = None
        self.softmax_list_reshape = None
        self.gen_loss_adv = None
        self.sample_word_list = None
        self.sample_word_list_reshape = None

    def build_input(self, name):
        """
        Build input placeholder
        Input
            :param name: name of network
        Output
            self.input_seqs_pre (if name == pretrain)
            self.input_seqs_mask (if name == pretrain, optional mask for masking invalid token)
            self.input_seqs_adv (if name == 'adversarial')
            self.reward (if name == 'adversarial')
        """
        assert name in ['pretrain', 'adversarial', 'sample']
        if name == 'pretrain':
            self.input_seqs_pre = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_seqs_pre')
            self.input_seqs_mask = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_seqs_mask')
        elif name == 'adversarial':
            self.input_seqs_adv = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_seq_adv')
            self.rewards = tf.placeholder(tf.float32, [None, self.sequence_length], name='reward')

    def build_pretrain_network(self):
        """
        Build pretrain network
        Input:
            self.input_seqs_pre
            self.input_seqs_mask
        Output:
            self.pretrain_loss
            self.pretrain_loss_sum(optional)
        """
        self.build_input(name='pretrain')
        self.pretrain_loss = 0.0
        state = None

        with tf.variable_scope('teller'):
            with tf.variable_scope('lstm'):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.device('/cpu:0'), tf.variable_scope('embedding'):
                word_emb_W = tf.get_variable('word_emb_W', [self.num_emb, self.emb_dim], tf.float32, self.initializer)
            with tf.variable_scope('output'):
                output_W = tf.get_variable('output_W', [self.emb_dim, self.num_emb], tf.float32, self.initializer)

            with tf.variable_scope('lstm'):
                for j in range(self.sequence_length):
                    with tf.device('/cpu:0'):
                        if j == 0:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.input_seqs_pre[:, j - 1])
                    if j == 0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)

                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())

                    logits = tf.matmul(output, output_W)
                    # calculate loss
                    pretrain_loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_seqs_pre[:, j],
                                                                                     logits=logits)
                    pretrain_loss_t = tf.reduce_sum(tf.multiply(pretrain_loss_t, self.input_seqs_mask[:, j]))
                    self.pretrain_loss += pretrain_loss_t
                    word_predict = tf.to_int32(tf.argmax(logits, 1))
            self.pretrain_loss /= tf.reduce_sum(self.input_seqs_mask)
            self.pretrain_loss_sum = tf.summary.scalar('pretrain_loss', self.pretrain_loss)

    def build_adversarial_network(self):
        """
        Build adversarial network
        Input:
            self.input_seqs_adv
            self.rewards
        Output:
            self.gen_loss_adv
        """
        self.build_input(name='adversarial')
        self.softmax_list = []
        self.softmax_list_reshape = []
        state = None

        with tf.variable_scope('teller'):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('lstm'):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.device('/cpu:0'), tf.variable_scope('embedding'):
                word_emb_W = tf.get_variable('word_emb_W', [self.num_emb, self.emb_dim], tf.float32, self.initializer)
            with tf.variable_scope('output'):
                output_W = tf.get_variable('output_W', [self.emb_dim, self.num_emb], tf.float32, self.initializer)

            with tf.variable_scope('lstm'):
                for j in range(self.sequence_length):
                    tf.get_variable_scope().reuse_variables()
                    with tf.device('/cpu:0'):
                        if j == 0:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.input_seqs_adv[:, j])
                    if j == 0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())

                    logits = tf.matmul(output, output_W)
                    softmax = tf.nn.softmax(logits)
                    self.softmax_list.append(softmax)

            self.softmax_list_reshape = tf.transpose(self.softmax_list, perm=[1, 0, 2])
            self.gen_loss_adv = -tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.input_seqs_adv, [-1])), self.num_emb, 1.0, 0.0) *
                    tf.log(tf.clip_by_value(tf.reshape(self.softmax_list_reshape, [-1, self.num_emb]), 1e-20, 1.0))
                    , 1) * tf.reshape(self.rewards, [-1])
            )

    def build_sample_network(self):
        """
        Build sample network
        Output
            self.sample_word_list_reshape
        """
        self.build_input('sample')
        self.sample_word_list = []
        state = None
        sample_word = None

        with tf.variable_scope('teller'):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope('lstm'):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.device('cpu:0'), tf.variable_scope('embedding'):
                word_emb_W = tf.get_variable('word_emb_W', [self.num_emb, self.emb_dim], tf.float32, self.initializer)
            with tf.variable_scope('output'):
                output_W = tf.get_variable('output_W', [self.emb_dim, self.num_emb], tf.float32, self.initializer)

            with tf.variable_scope('lstm'):
                for j in range(self.sequence_length):
                    tf.get_variable_scope().reuse_variables()
                    with tf.device('/cpu:0'):
                        if j == 0:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, self.start_token)
                        else:
                            lstm1_in = tf.nn.embedding_lookup(word_emb_W, sample_word)
                    if j == 0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)
                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())
                    logits = tf.matmul(output, output_W)
                    logprob = tf.log(tf.nn.softmax(logits))
                    sample_word = tf.reshape(tf.to_int32(tf.multinomial(logprob, 1)), [self.batch_size])
                    self.sample_word_list.append(sample_word)
            self.sample_word_list_reshape = tf.transpose(
                tf.squeeze(tf.stack(self.sample_word_list)),
                perm=[1, 0]
            )

    def build(self):
        """
        Create all network for pre-training, adversarial training and sampling
        """
        self.build_pretrain_network()
        self.build_adversarial_network()
        self.build_sample_network()
    def generate(self,sess):
        """
        Helper function for sample generation
        """
        sess.run(self.sample_word_list_reshape)
