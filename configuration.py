class generator_config(object):
    """
    Wrapper class for generator hyper parameters
    """

    def __init__(self):
        self.start_token = 0  # special token index for start of sequence


class discriminator_config(object):
    """
    Wrapper class for discriminator hyper parameters
    """

    def __init__(self):
        self.sequence_length = 20  # maximum input sequence length


class training_config(object):
    """
    Wrapper class for training hyper parameters
    """

    def __init__(self):
        self.gen_learning_rate = 0.0001  # learning rate of generator
