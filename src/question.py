import os
import time
import tqdm

from data.Latent import *
from logger.LoggerFacade import LoggerFacade
from models.qass import Vaios_model
from networks.ConvNew32 import ConvNew32
from networks.DeconvNew32 import DeconvNew32
from models.CostType import CostType
from data.DatasetFacade import DatasetFacade
from data.Mnist32 import Mnist32

from networks.DenseGenerator import *
from networks.DenseCritic import *
from Settings import Settings

data_path = os.getcwd() + os.sep + "Data"
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class AssignmentTraining_new_approach:

    def __init__(self, generator_network=None,
                 critic_network=None,
                 dataset=None,
                 latent=None,
                 ):


        self.dataset = dataset
        self.latent = latent
        self.critic_network = critic_network
        self.generator_network = generator_network
        self.experiment_name = "log_" + time.strftime("%Y-%m-%d_%H-%M-%S_")
        self.model = Vaios_model(self.dataset, self.latent, self.generator_network, self.critic_network)
        self.global_step = tf.train.get_or_create_global_step(graph=None)
        self.increase_global_step = tf.assign_add(self.global_step, 1)
        self.session = Settings.create_session()
        self.session.run(tf.initialize_all_variables())


    def train(self, n_critic_loops=None, n_main_loops=None, n_assign_loops=None):
        with self.session as session:
            log_path = os.path.join(os.getcwd(), "logs" + os.sep + self.experiment_name)
            log_writer = tf.summary.FileWriter(log_path, session.graph)
            self.logger = LoggerFacade(session, log_writer, self.dataset.shape)
            global_step = session.run(self.global_step)
            self.n_zeros = self.latent.batch_size
            assign_training = True
            for main_loop in tqdm.tqdm(range(global_step, n_main_loops, 1)):
                latent_sample_big = []
                real_idx_big = []
                with tqdm.tqdm(range(n_critic_loops)) as crit_bar:
                    if assign_training:
                        for crit_loop in crit_bar:
                            assign_arr,latent_sample,real_idx = self.model.find_assignments_critic(session, assign_loops=n_assign_loops)
                            self.model.train_critic(session, assign_arr)
                            self.n_zeros = len(assign_arr) - np.count_nonzero(assign_arr)
                            latent_sample_big=latent_sample_big+latent_sample
                            real_idx_big=real_idx_big+real_idx
                            crit_bar.set_description(
                                "step 1: # zeros " + str(self.n_zeros) + " Variance" + str(np.var(assign_arr)),
                                refresh=False)
                latent_sample_big = np.vstack(tuple(latent_sample_big))
                real_idx_big = np.vstack(tuple(real_idx_big)).flatten()
                self.model.train_generator(session, real_idx_big,latent_sample_big,offset=200)

                session.run(self.increase_global_step)
                fakes = session.run(self.model.get_fake_tensor(), {self.model.latent_batch_ph: latent_sample_big[150:168]})
                reals = self.dataset.data[real_idx_big[150:168]]
                self.log_data(main_loop,n_main_loops)
                self.logger.log_image_grid_fixed(fakes, reals, main_loop, name="real_and_assigned")
            log_writer.close()

    def log_data(self, main_loop,max_loop):

        # accumulate some real and fake samples
        if max_loop-1 == main_loop:
            fake_points = self.session.run(self.model.fake_samples)
            dataset_dim = self.dataset.get_total_shape()
            n_fake_to_save = 100000
            while(fake_points.shape[0]<n_fake_to_save):
                fake_points_new = self.session.run(self.model.fake_samples)
                fake_points = np.vstack((fake_points, fake_points_new))
            dump_path =  "logs" + os.sep + self.experiment_name+os.sep
            np.save(dump_path + "fakes", fake_points)


def main():
    Settings.setup_enviroment(gpu=0)
    assignment_training = AssignmentTraining_new_approach(dataset=Mnist32(batch_size=300,dataset_size=300),
                                                          latent=VaiosLatent2(shape=20, batch_size=200),
                                                          critic_network=DenseCritic(name="critic", learn_rate=5e-5,
                                                                                   layer_dim=512,xdim=32*32),
                                                          generator_network=DenseGenerator(name="generator",
                                                                                        learn_rate=1e-5, layer_dim=512,xdim=32*32)
                                                          )
    assignment_training.train(n_main_loops=1000, n_critic_loops=10, n_assign_loops=5)

if __name__ == "__main__":
    main()
