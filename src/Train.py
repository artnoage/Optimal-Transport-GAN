import os
import time
import tqdm

from data.Latent import *
from logger.LoggerFacade import LoggerFacade
from models.Assignment_model import Assignment_model
from networks.Conv32 import Conv32
from networks.Deconv32 import Deconv32
from data.DatasetFacade import DatasetFacade
from data.Mnist32 import Mnist32
from data.Fashion32 import Fashion32
from data.Cifar import Cifar
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
        self.experiment_name = self.dataset.name + "_"\
                               + self.cost + "_" \
                               + self.latent.name+ "_" \
                               + time.strftime("_%Y-%m-%d_%H-%M-%S_")
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
            self.n_non_assigned = self.latent.batch_size
            for main_loop in tqdm.tqdm(range(global_step, n_main_loops, 1)):

                #The ideal number for assigment points is 10-12 times the size of the dataset. However it is convinient
                # to increase the size with the iterations because in the beginning of the training, big number of assignments does
                #not help that much

                assignment_loops=int(10*(self.dataset.dataset_size/self.latent.batch_size)*np.sqrt(main_loop/n_main_loops))+10
                with tqdm.tqdm(range(n_critic_loops)) as crit_bar:
                    for crit_loop in crit_bar:
                        assign_arr, latent_sample,real_idx = self.model.find_assignments_critic(session, assign_loops= assignment_loops)
                        self.model.train_critic(session, assign_arr)
                        self.n_non_assigned = len(assign_arr) - np.count_nonzero(assign_arr)
                        crit_bar.set_description(
                        "step 1: Number of non assigned points " + str(self.n_non_assigned) + ",  Variance from perfect assignment" + str(np.var(assign_arr)),
                            refresh=False)

                latent_sample = np.vstack(tuple(latent_sample))
                real_idx = np.vstack(tuple(real_idx)).flatten()

                # The smaller the offset the more precisely the generator learns. However very small offset number
                # increases the training a lot and may lead to difficulties of training the
                # critic because it converges fast to some of the nodes.

                self.model.train_generator(session, real_idx,latent_sample,offset=16)

                session.run(self.increase_global_step)

                # It makes images for Tensorboard
                if main_loop%50==1:
                    latent_sample_noisy = latent_sample[:18] + np.random.normal(loc=0, scale=0.3, size=(18, self.latent.shape))
                    latent_sample2 = np.copy(latent_sample[:18])
                    np.random.shuffle(latent_sample2)
                    int_sample = 0.8 * latent_sample[:18] + 0.2 * latent_sample2

                    reals = self.dataset.data[real_idx[:18]]
                    fakes = session.run(self.model.get_fake_tensor(), {self.model.latent_batch_ph: latent_sample[:18]})
                    fakes_noisy = session.run(self.model.get_fake_tensor(),
                                              {self.model.latent_batch_ph: latent_sample_noisy})
                    fakes_int = session.run(self.model.get_fake_tensor(), {self.model.latent_batch_ph: int_sample})

                    self.log_data(main_loop, n_main_loops, session)
                    self.logger.log_image_grid_fixed(fakes, reals, main_loop, name="real_and_assigned")
                    # This is for the case where one uses multiGaussian as a way to sample the latent space. Since in
                    # case we have a perfect fit, if one want to see "new" generated points has to move a bit further from
                    # the Gaussians

                    #self.logger.log_image_grid_fixed(fakes, fakes_noisy, main_loop, name="Generated_and_neighbours.")

                    #This is to see how points between two generated points look like

                    #self.logger.log_image_grid_fixed(fakes, fakes_int, main_loop, name="Interpolations_between_generated")
            log_writer.close()

    def log_data(self, main_loop,max_loop,session):
        # accumulate fake samples and dump them into a file at the end of the training
        if max_loop-1 == main_loop:
            latent_points = session.run(self.model.generate_latent_batch)
            fake_points = session.run(self.model.get_fake_tensor(), {self.model.latent_batch_ph: latent_points})
            n_fake_samples_to_save = 100000
            while(fake_points.shape[0]<n_fake_samples_to_save):
                latent_points = session.run(self.model.generate_latent_batch)
                fake_points_new = session.run(self.model.get_fake_tensor(), {self.model.latent_batch_ph: latent_points})
                fake_points = np.vstack((fake_points, fake_points_new))
            dump_path =  "logs" + os.sep + self.experiment_name+os.sep
            np.save(dump_path + "fakes_" +str(main_loop), fake_points)

#For cost you have the options "sqaure", "psnr" and "ssim". "Psnr option trains only the critic with psnr and the generator with SSIM.
# The reason for doing that is that training the critic with ssim is computationally very expensive while with psnr is cheap and the results are quite similar.
# Note that psnr is not good enough to train the generator and that is why we dont include an option where psnr is used for both networks.
# For the batch_sizes try to have the Real points batch size as close to the dataset_size as possible. Then increase latent batch size to the degree that memory
# allows
def main():
    Settings.setup_enviroment(gpu=0)
    assignment_training = AssignmentTraining(dataset=Fashion32(batch_size=1000, dataset_size=1000),
                                             latent=MultiGaussian_latent(shape=250, batch_size=200),
                                             critic_network=DenseCritic(name="critic", learn_rate=5e-5,layer_dim=1024,xdim=32*32*1),
                                             generator_network=Deconv32(name="generator", learn_rate=1e-4, layer_dim=512),
                                             cost="square")
    assignment_training.train(n_main_loops=200, n_critic_loops=10)

if __name__ == "__main__":
    main()