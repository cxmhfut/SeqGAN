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
