import json
import os

import numpy as np
import tensorflow as tf
from kvae.models import KalmanMixture
from kvae.utils import reload_config, get_2d_config

np.random.seed(1337)


def run():
    """Load and train model

    Create a model object and run the training using the provided config. The saving of the
    config for some reason needs to be done here and not in the model class
    """
    config = get_2d_config()
    config = reload_config(config.FLAGS)

    # Add timestamp to log path
    config.log_dir = os.path.join(config.log_dir, '%s' % config.run_name)

    # Create log path
    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)

    # Save hyperparameters
    with open(config.log_dir + '/config.json', 'w') as f:
        json.dump(config.__flags, f)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    with tf.Session() as sess:
        model = KalmanMixture(config, sess)
        model.build_model().build_loss().initialize_variables()
        return model.train()

if __name__ == "__main__":
    lb = run()
