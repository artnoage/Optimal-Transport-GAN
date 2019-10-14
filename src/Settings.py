import os
import tensorflow as tf
import sys

class Settings:

    @staticmethod
    def setup_enviroment(gpu=1,disable_cache=True):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

        if disable_cache:
            os.environ['CUDA_CACHE_DISABLE'] = '1'
        tf.reset_default_graph()
        sys.path.append(os.getcwd())

    @staticmethod
    def create_session():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)


