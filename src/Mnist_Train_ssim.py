import os
import time
import tqdm

from data.Latent import *
from logger.LoggerFacade import LoggerFacade
from models.Assignment_model import Assignment_model
from networks.ConvNew32 import ConvNew32
from networks.DeconvNew32 import DeconvNew32
from data.DatasetFacade import DatasetFacade
from data.Mnist import Mnist32
from data.Fashion import Fashion32
from networks.DenseGenerator import *
from networks.DenseCritic import *
from Settings import Settings


data_path = os.getcwd() + os.sep + "Data"
import numpy as np


class AssignmentTraining:

    def __init__(self, generator_network=None,
                 critic_network=None,
                 dataset=None,
                 latent=None,
                 cost=None):

        self.dataset = dataset
        self.latent = latent
        self.critic_network = critic_network
        self.generator_network = generator_network
        self.cost=cost
        self.experiment_name = self.dataset.name + "_" + self.cost + "_" + self.latent.name+ "_" + time.strftime("_%Y-%m-%d_%H-%M-%S_")
        self.model = Assignment_model(self.dataset, self.latent, self.generator_network, self.critic_network,self.cost)
        self.global_step = tf.train.get_or_create_global_step(graph=None)
        self.increase_global_step = tf.assign_add(self.global_step, 1)
        self.session = Settings.create_session()
        self.session.run(tf.initialize_all_variables())


    def train(self, n_critic_loops=None, n_main_loops=None):
        with self.session as session:
            log_path = os.path.join(os.getcwd(), "logs" + os.sep + self.experiment_name)
            log_writer = tf.summary.FileWriter(log_path, session.graph)
            self.logger = LoggerFacade(session, log_writer, self.dataset.shape)
            global_step = session.run(self.global_step)
            self.n_zeros = self.latent.batch_size
            for main_loop in tqdm.tqdm(range(global_step, n_main_loops, 1)):
                with tqdm.tqdm(range(n_critic_loops)) as crit_bar:
                    for crit_loop in crit_bar:
                        assign_arr, latent_sample,real_idx = self.model.find_assignments_critic(session, assign_loops=int(15*self.dataset.dataset_size/self.latent.batch_size))
                        self.model.train_critic(session, assign_arr)
                        self.n_zeros = len(assign_arr) - np.count_nonzero(assign_arr)
                        crit_bar.set_description(
                        "step 1: # zeros " + str(self.n_zeros) + " Variance" + str(np.var(assign_arr)),
                            refresh=False)
                if self.n_zeros!=0:
                    latent_sample = np.vstack(tuple(latent_sample))
                    real_idx = np.vstack(tuple(real_idx)).flatten()
                    self.model.train_generator(session, real_idx,latent_sample,offset=64)
                if self.n_zeros==0:
                    assign_arr, latent_sample, real_idx = self.model.find_assignments_critic(session, assign_loops=100)
                    latent_sample = np.vstack(tuple(latent_sample))
                    real_idx = np.vstack(tuple(real_idx)).flatten()
                    self.model.train_generator(session, real_idx,latent_sample,offset=64)

                session.run(self.increase_global_step)

                # It makes images for Tensorboard

                fakes = session.run(self.model.get_fake_tensor(), {self.model.latent_batch_ph: latent_sample[:18]})
                reals = self.dataset.data[real_idx[:18]]
                self.log_data(main_loop,n_main_loops)
                self.logger.log_image_grid_fixed(fakes, reals, main_loop, name="real_and_assigned")
            log_writer.close()

    def log_data(self, main_loop,max_loop):

        # accumulate some real and fake samples
        if max_loop-1 == main_loop or main_loop==int(max_loop/2):
            fake_points = self.session.run(self.model.generate_fake_samples)
            n_fake_to_save = 100000
            while(fake_points.shape[0]<n_fake_to_save):
                fake_points_new = self.session.run(self.model.generate_fake_samples)
                fake_points = np.vstack((fake_points, fake_points_new))
            dump_path =  "logs" + os.sep + self.experiment_name+os.sep
            np.save(dump_path + "fakes_" +str(max_loop), fake_points)


def main():
    Settings.setup_enviroment(gpu=0)
    assignment_training = AssignmentTraining(dataset=Mnist32(batch_size=100, dataset_size=100),
                                             latent=Assignment_latent(shape=250, batch_size=800),
                                             critic_network=DenseCritic(name="critic", learn_rate=1e-4,
                                                                                   layer_dim=512,xdim=32*32),
                                             generator_network=DeconvNew32(name="generator",
                                                                                        learn_rate=1e-4, layer_dim=512),
                                             cost="ssim")
    assignment_training.train(n_main_loops=100, n_critic_loops=10)

if __name__ == "__main__":
    main()