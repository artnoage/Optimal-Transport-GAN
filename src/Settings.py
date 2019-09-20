import os
import tensorflow as tf
import sys

class Settings:

    @staticmethod
    def setup_enviroment(gpu=1,disable_cache=True):
        #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        if gpu == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if gpu == 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        if gpu == 2:
            os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        if gpu == 3:
            os.environ['CUDA_VISIBLE_DEVICES'] = '3'

        if disable_cache:
            os.environ['CUDA_CACHE_DISABLE'] = '1'
        tf.reset_default_graph()
        sys.path.append(os.getcwd())

    @staticmethod
    def create_session():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)


