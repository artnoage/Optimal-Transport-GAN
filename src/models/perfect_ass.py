import tensorflow as tf
import numpy as np

class Vaios_model:

    def __init__(self,dataset,latent,generator,critic,A_couples=None,A_cost=None):
        self.dataset = dataset
        self.latent = latent
        self.gen_network = generator
        self.crit_network = critic
        self.A_couples = A_couples
        self.A_cost = A_cost
        self.define_graph()


    def define_graph(self):

        self.z = self.latent.tensor()
        self.dataset_size = self.dataset.dataset_size
        self.latent_batch_ph = tf.placeholder(shape=(None, self.latent.shape), dtype=tf.float32, name="latent_batch_ph")
        self.fake_samples = self.gen_network.tensor(self.z, self.dataset.get_total_shape())
        self.real_batch_ph=tf.placeholder(shape=(None, self.dataset.get_total_shape()), dtype=tf.float32, name="real_batch_ph")
        self.crit_fake = self.crit_network.tensor(self.fake_samples,xdim=512)
        self.crit_real_ph = tf.placeholder(tf.float32,name="crit_real")
        self.best_idx_ph = tf.placeholder(tf.int32, shape=(None,), name="best_idx")
        self.index_total_size_ph = tf.placeholder(tf.int32, shape=(), name="index_total_size")
        self.assign_samples_ph = tf.placeholder(tf.float32, shape=(None, self.dataset.get_total_shape()))
        self.n_assign_ph = tf.placeholder(tf.float32, shape=(None,),name="assign")
        self.crit_train_op = self.crit_network.train_op(self.assignment_critic_cost())
        self.find_couples= self.find_couples_unlimited_ssim()
        self.fake_batch = self.gen_network.tensor(self.latent_batch_ph, self.dataset.get_total_shape())
        self.critic_cost=self.assignment_critic_cost()
        self.gen_cost = self.assignment_generator_cost_ssim()
        self.gen_train_op = self.gen_network.train_op(self.gen_cost)

    def generate_samples(self,session,latent_sample):
        return session.run(self.fake_batch, feed_dict={self.latent_batch_ph: latent_sample[0:5000]})

    def get_fake_tensor(self):
        return self.fake_batch

    def find_assignments_critic(self,session, assign_loops=100):
        ratio = int(self.dataset_size / self.dataset.batch_size)
        assign_arr = np.zeros(shape=(self.dataset_size,))
        latent_sample_list, real_idx_list = [], []
        for assign_loop in range(assign_loops):
            latent_points=session.run(self.z)
            for ith_batch in range(ratio):
                index =np.arange(ith_batch * self.dataset.batch_size, (ith_batch + 1) * self.dataset.batch_size)
                if ith_batch==0:
                    all_index=index
                if ith_batch!=0:
                    all_index=np.hstack((current_best,index))
                current_best = session.run(self.find_couples, feed_dict={self.real_batch_ph: self.dataset.data[all_index], self.latent_batch_ph:latent_points})
                current_best=all_index[current_best]
            assign_c = np.reshape(current_best, newshape=[-1, 1])
            latent_sample_list.append(latent_points)
            real_idx_list.append(assign_c)
            idx_value = np.unique(assign_c, return_counts=True)
            assign_arr[idx_value[0]] += idx_value[1]
        return assign_arr,latent_sample_list,real_idx_list

    def find_couples_unlimited_ssim(self):
        fake_samples= self.fake_samples
        latent_batch_size = self.latent.batch_size
        single_image_shape = self.dataset.shape
        fake_samples_shape = (-1,) + single_image_shape
        real_matrix_shape = (-1,) + single_image_shape
        real_points =self.real_batch_ph
        new1 = tf.keras.backend.repeat_elements(tf.reshape(real_points,real_matrix_shape),latent_batch_size,axis=0)
        new2 = tf.tile(tf.reshape(fake_samples, fake_samples_shape), [tf.shape(real_points)[0], 1, 1, 1])
        new1 = new1 + 1
        new2 = new2 + 1
        dist = 1 - tf.image.psnr(
            new1,
            new2,
            2
        )

        dist = tf.transpose(self.crit_network.tensor(real_points)) + tf.transpose(tf.reshape(dist, (tf.shape(real_points)[0], tf.shape(fake_samples)[0])))
        couples = tf.argmin(dist, axis=1, output_type=tf.int32)
        return couples

    def find_couples_unlimited_square(self):
        real_points = self.real_batch_ph
        z = tf.expand_dims(self.fake_samples, axis=1) - real_points
        norm_mat = 0.1 * tf.square(tf.norm(z, axis=2))
        dist = tf.transpose(self.crit_network.tensor(real_points)) + norm_mat
        couples=tf.argmin(dist, axis=1, output_type=tf.int32)
        return couples



    def assignment_critic_cost(self):
        crit_assign = self.crit_network.tensor(self.assign_samples_ph,xdim=512)
        crit_assign_weighted = tf.multiply(self.n_assign_ph, tf.squeeze(crit_assign))
        crit_cost = -(
                (tf.reduce_sum(crit_assign_weighted) / tf.reduce_sum(self.n_assign_ph))
                - tf.reduce_mean(self.crit_network.tensor(self.real_batch_ph,xdim=512)))
        return crit_cost

    def train_critic(self,session,assign_arr):
        assign_idx_local = np.nonzero(assign_arr)
        samples = {}
    #    a=np.random.choice(range(self.dataset.dataset_size), 2*self.dataset.batch_size, replace=False)
        samples.update({self.crit_network.get_training_placeholder():True})
        samples.update({self.assign_samples_ph: self.dataset.data[assign_idx_local],
                        self.n_assign_ph: assign_arr[assign_idx_local],
                        self.real_batch_ph: self.dataset.data[np.arange(0,self.dataset.dataset_size)]
                        })
        _,A=session.run([self.crit_train_op,self.critic_cost], samples)

    def assignment_generator_cost_ssim(self):
        single_image_shape = self.dataset.shape
        batch_image_shape = (-1,) + single_image_shape
        self.real_batch = self.real_batch_ph
        return tf.reduce_mean(
            1-tf.image.ssim(tf.reshape(self.real_batch,batch_image_shape)+1 ,
                          tf.reshape(self.fake_batch,batch_image_shape)+1
                          ,2


            )
        )
    def assignment_generator_cost_square(self):
        subed_batches = self.fake_batch - self.real_batch_ph
        gen_cost = tf.reduce_mean(tf.square(tf.norm(subed_batches, axis=0)))
        return gen_cost



    def train_generator(self, session, real_idx, latent_sample, offset=2000):
        for c_idx in range(0, int(len(real_idx) - offset + 1), int(offset)):
            step_2_dict = {}
            step_2_dict.update({self.gen_network.get_training_placeholder(): True})
            step_2_dict.update({self.real_batch_ph: self.dataset.data[real_idx[c_idx:c_idx + offset]],
                                self.latent_batch_ph: latent_sample[c_idx:c_idx + offset]})
            session.run([self.gen_train_op], step_2_dict)
