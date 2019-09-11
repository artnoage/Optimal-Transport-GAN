from abc import ABC


class ModelsABC(ABC):

    def __init__(self):
        self.dataset = None
        self.latent = None
        self.gen_network = None
        self.crit_network = None

    def train_discriminator(self,session):
        """
        Trains the  discriminator or the critic
        :param session: the tensorflow  session to use
        :return:
        """
        pass

    def train_generator(self,session):
        """
        Trains the generator
        :param session: the tensorflow  session to use
        :return:
        """
        pass

    def generate_samples(self,session):
        """
        Generate samples from the generator

        :param session: the tensorflow  session to use
        :return: generated samples
        """
        pass


