import tensorflow as tf


class DatasetFacade:
    """
        A Facade to give an easy interface to getting a working feed dict
    """

    def __init__(self, dataset, latent,
                 real_samples_ph, latent_samples_ph):
        self.dataset = dataset
        self.latent = latent
        self.real_samples_ph = real_samples_ph
        self.latent_samples_ph = latent_samples_ph
        self.feed_dict = None

    def get_data_dict(self, n_real_samples=None, n_latent_samples=None, training=[]):
        """
        Returns a feed_dict for evaluation parts of the tensorboard graph.
        The dictionary includes variables for activating train and testing of the batchnorm and
        a constant that can be used for experiments.
        If one batch per iteration is active and a batch was already used this iteration the same
        batch is returned each time.

        :param n_real_samples:  How many real samples we want if not set the default size is used
        :param n_latent_samples: How many latent samples we want if not set the default size is used
        :param critic_train_phase: Boolean if the critic is being trained.
        :param generator_train_phase: Boolean if the generator is being trained.
        :return: feed_dict to train the networks
        """
        feed_dict = {}
        for network_name in training:
            feed_dict.update(self.train_placeholder(network_name))
        # generate new samples everytime
        feed_dict.update({
            self.real_samples_ph: self.dataset.sample(batch_size=n_real_samples),
            self.latent_samples_ph: self.latent.sample(batch_size=n_latent_samples)}
        )
        return feed_dict

    @staticmethod
    def train_placeholder(network_name):
        """
        Returns a dict that sets the training placeholder for the network specified in the param network_name to True

        :param network_name: the name of the network
        :return: a dict with the corresponding training placeholder set to true
        """
        feed_dict = {}
        placeholder = tf.get_default_graph().get_tensor_by_name(network_name + 'is_training' + ':0')
        feed_dict.update({placeholder: True})
        return feed_dict
