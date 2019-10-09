import tensorflow as tf
import numpy as np

class Assignment_model:

    def __init__(self,dataset,latent,generator,critic,cost,A_couples=None,A_cost=None):
        self.cost=cost
        self.dataset = dataset
        self.latent = latent
        self.gen_network = generator
        self.crit_network = critic
        self.A_couples = A_couples
        self.A_cost = A_cost
        # For tf.image.ssim_multiscale we only provide two weights because cifar can't be downsampled more than
        # one time without getting to smal for a filter size of 11
        # The orgignal weights are  (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
        self.power_factors = (0.0448, 0.2856)
        self.define_graph()


    def define_graph(self):

        # tensor that will generate a new latent batch each time we use it in the graph
        self.generate_latent_batch = self.latent.tensor()

        # placeholder that allows us to feed a latent tensor into the graph
        self.latent_batch_ph = tf.placeholder(shape=(None, self.latent.shape), dtype=tf.float32)

        # output from the generator for a feeded latent space and a random one.
        self.generate_fake_samples = self.gen_network.tensor(self.generate_latent_batch, self.dataset.get_total_shape())
        self.feeded_generated_batch = self.gen_network.tensor(self.latent_batch_ph, self.dataset.get_total_shape())

        # placeholder to feed a batch of images into the graph
        self.real_batch_ph = tf.placeholder(shape=(None, self.dataset.get_total_shape()), dtype=tf.float32, name="real_batch_ph")

        # critic output of generated fake samples
        self.crit_fake = self.crit_network.tensor(self.generate_fake_samples)

        # placeholder to directly feed the output of the critic into the network to save computations
        self.crit_real_ph = tf.placeholder(tf.float32,name="crit_real")

        # we feed the current best assignments into the graph to iterate over bigger datasets
        self.best_idx_ph = tf.placeholder(tf.int32, shape=(None,), name="best_idx")

        # we feed the assignments into the networks and how many times the assignments happend.
        # this reduces computational effort if we calculate assignments multiple times
        self.assign_samples_ph = tf.placeholder(tf.float32, shape=(None, self.dataset.get_total_shape()))
        self.n_assign_ph = tf.placeholder(tf.float32, shape=(None,))

        if self.cost=="ssim":
            self.find_couples=self.find_couples_unlimited_ssim()
            self.gen_cost = self.assignment_generator_cost_ssim()

        if self.cost=="psnr":
            self.find_couples=self.find_couples_unlimited_psnr()
            self.gen_cost = self.assignment_generator_cost_ssim()

        if self.cost=="square":
            self.find_couples = self.find_couples_unlimited_square()
            self.gen_cost = self.assignment_generator_cost_square()


        self.gen_train_op = self.gen_network.train_op(self.gen_cost)
        self.crit_train_op = self.crit_network.train_op(self.assignment_critic_cost())

    def generate_samples(self,session,latent_sample):
        return session.run(self.feeded_generated_batch, feed_dict={self.latent_batch_ph: latent_sample[0:5000]})

    def get_fake_tensor(self):
        return self.feeded_generated_batch

    def find_assignments_critic(self,session, assign_loops=100):
        ratio = int(self.dataset.dataset_size / self.dataset.batch_size)
        assign_arr = np.zeros(shape=(self.dataset.dataset_size,))
        latent_sample_list, real_idx_list = [], []
        for assign_loop in range(assign_loops):
            latent_points=session.run(self.generate_latent_batch)
            for ith in range(ratio):
                index=np.arange(ith*self.dataset.batch_size,(ith+1)*self.dataset.batch_size)
                if ith==0:
                    all_index=index
                if ith!=0:
                    all_index=np.concatenate([current_best,index],axis=0)
                best = session.run(self.find_couples,feed_dict={self.real_batch_ph:self.dataset.data[all_index],self.latent_batch_ph:latent_points})
                current_best = all_index[best]
            assign_c = np.reshape(current_best, newshape=[-1, 1])
            latent_sample_list.append(latent_points)
            real_idx_list.append(assign_c)
            idx_value = np.unique(assign_c, return_counts=True)
            assign_arr[idx_value[0]] += idx_value[1]
        return assign_arr,latent_sample_list,real_idx_list

    def find_couples_unlimited_ssim(self):
        fake_samples= self.feeded_generated_batch
        latent_batch_size = self.latent.batch_size
        single_image_shape = self.dataset.shape
        fake_samples_shape = (-1,) + single_image_shape
        real_matrix_shape = (-1,) + single_image_shape
        real_points = self.real_batch_ph

        # ssim and multiscale ssim tensorflow implementation compare image batches against image batches.
        # To calculate the distance matrix between real samples and fake samples we create batches of reals and
        # fakes that calculate each entry in the matrix
        new1 = tf.keras.backend.repeat_elements(tf.reshape(real_points, real_matrix_shape), latent_batch_size, axis=0)
        new2 = tf.tile(tf.reshape(fake_samples, fake_samples_shape), [tf.shape(real_points)[0], 1, 1, 1])

        # the ssim implementation needs positive values we shift by one
        new1 = new1 + 1
        new2 = new2 + 1

        # if we have color images we use ssim_multiscale
        # maximum pixel values go from 0 to 2
        if self.dataset.shape[2] == 3:
            dist = 5-5*tf.image.ssim_multiscale(new1,new2,2,self.power_factors)
        else:
            dist = 5 - 5*tf.image.ssim(new1,new2,2)

        # we reshape the calucated distance vector to be a distance matrix
        dist = tf.transpose(self.crit_network.tensor(real_points)) + tf.transpose(
            tf.reshape(dist, (tf.shape(real_points)[0], tf.shape(fake_samples)[0])))
        # we find the minimal distance between a fake sample and a real one and return the index position of the real
        couples  = tf.argmin(dist, axis=1, output_type=tf.int32)
        return couples

    def find_couples_unlimited_psnr(self):
        fake_samples = self.feeded_generated_batch
        latent_batch_size = self.latent.batch_size
        single_image_shape = self.dataset.shape
        fake_samples_shape = (-1,) + single_image_shape
        real_matrix_shape = (-1,) + single_image_shape
        real_points = self.real_batch_ph

        # psnr ssim tensorflow implementation compare image batches against image batches.
        # To calculate the distance matrix between real samples and fake samples we create batches of reals and
        # fakes that calculate each entry in the matrix
        new1 = tf.keras.backend.repeat_elements(tf.reshape(real_points, real_matrix_shape), latent_batch_size, axis=0)
        new2 = tf.tile(tf.reshape(fake_samples, fake_samples_shape), [tf.shape(real_points)[0], 1, 1, 1])
        # shifting might not be needed for psnr but also dosen't hurt
        new1 = new1 + 1
        new2 = new2 + 1
        dist = 1 - tf.image.psnr(
            new1,
            new2,
            2)

        dist = tf.transpose(self.crit_network.tensor(real_points)) + tf.transpose(
            tf.reshape(dist, (tf.shape(real_points)[0], tf.shape(fake_samples)[0])))
        # we find the minimal distance between a fake sample and a real one and return the index position of the real
        couples = tf.argmin(dist, axis=1, output_type=tf.int32)
        return couples



    def find_couples_unlimited_square(self):
        fake_samples = self.feeded_generated_batch
        real_points = self.real_batch_ph
        # we use broadcasting to get a matrix of all fakes - each real
        z = tf.expand_dims(fake_samples, axis=1) - real_points
        # calculating the squared distance for the matrix
        norm_mat = 0.1 * tf.square(tf.norm(z, axis=2))
        dist=tf.transpose(self.crit_network.tensor(real_points)) + norm_mat

        # return the real positions where the distance is minimal
        couples = tf.argmin(dist, axis=1, output_type=tf.int32)
        return couples


    def assignment_critic_cost(self):
        # we save some computations by multiplying the critic results of multiple assignments happend
        crit_assign = self.crit_network.tensor(self.assign_samples_ph)
        crit_assign_weighted = tf.multiply(self.n_assign_ph, tf.squeeze(crit_assign))
        crit_cost = -(
                (tf.reduce_sum(crit_assign_weighted) / tf.reduce_sum(self.n_assign_ph))
                - tf.reduce_mean(self.crit_network.tensor(self.dataset.data[np.arange(0,self.dataset.dataset_size)])))
        return crit_cost


    def assignment_generator_cost_ssim(self):
        single_image_shape = self.dataset.shape
        batch_image_shape = (-1,) + single_image_shape
        self.real_batch = self.real_batch_ph
        if self.dataset.shape[2] == 3:
            # +1 to have positive pixel values
            gen_cost = 1-tf.image.ssim_multiscale(tf.reshape(self.real_batch,batch_image_shape) + 1,
                            tf.reshape(self.feeded_generated_batch, batch_image_shape) + 1,
                                                  2,
                                                  self.power_factors
            )
        else:
            # +1 to have positive pixel values
            gen_cost=tf.reduce_mean(1-tf.image.ssim(tf.reshape(self.real_batch,batch_image_shape) + 1,
                                                    tf.reshape(self.feeded_generated_batch, batch_image_shape) + 1
                                                    , 2
                                                    ,filter_size=4)
            )
        return gen_cost

    def assignment_generator_cost_square(self):
        self.real_batch = self.real_batch_ph
        subed_batches = self.feeded_generated_batch - self.real_batch
        gen_cost = tf.reduce_mean(tf.square(tf.norm(subed_batches, axis=1)))
        return gen_cost

    def train_critic(self,session,assign_arr):
        assign_idx_local = np.nonzero(assign_arr)
        samples = {}
        samples.update({self.crit_network.get_training_placeholder():True})
        samples.update({self.assign_samples_ph: self.dataset.data[assign_idx_local],
                        self.n_assign_ph: assign_arr[assign_idx_local]})
        session.run(self.crit_train_op, samples)

    def train_generator(self, session, real_idx, latent_sample, offset=2000):
        cost=[]
        for c_idx in range(0, int(len(real_idx) - offset + 1), int(offset)):
            step_2_dict = {}
            step_2_dict.update({ self.real_batch_ph: self.dataset.data[real_idx[c_idx:c_idx + offset]],
                                self.latent_batch_ph: latent_sample[c_idx:c_idx + offset]})
            step_2_dict.update({self.gen_network.get_training_placeholder(): True})
            _, A = session.run([self.gen_train_op, self.gen_cost], step_2_dict)
            cost.append(A)
        print("The trasportation distance is", np.average(A))