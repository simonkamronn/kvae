import tensorflow as tf
import time
import os
import json


def reload_config(FLAGS):
    # If we are reloading a model, overwrite the flags
    if FLAGS.reload_model is not '':
        with open('%s/%s' % (os.path.dirname(FLAGS.reload_model), 'config.json')) as data_file:
            config_dict = json.load(data_file)

        for key, value in config_dict.items():
            attr_remove = ['gpu', 'run_name', 'log_dir', 'n_steps_gen', 'reload_model', 'display_step', 'generate_step']
            # attr_remove = ['gpu', 'n_steps_gen', 'reload_model', 'display_step', 'generate_step']
            if key not in attr_remove:
                FLAGS.__setattr__(key, value)
    return FLAGS


def get_image_config():
    cl = tf.app.flags

    # Choose data set
    cl.DEFINE_string('dataset', 'box', 'Select data set') # 'box_rnd', 'box_gravity', 'polygon' or 'pong'

    # VAE config
    cl.DEFINE_string('out_distr', 'bernoulli', 'Output distibution')
    cl.DEFINE_boolean('conv', True, 'Use conv vae')
    cl.DEFINE_string('activation', 'relu', 'Activation function in VAE')
    cl.DEFINE_integer('filter_size', 3, 'Filter size in conv vae')
    cl.DEFINE_string('num_filters', '32,32,32', 'Comma separated list of conv filters')
    cl.DEFINE_integer('vae_num_units', 25, 'Number of hidden units in the VAE (if conv=False)')
    cl.DEFINE_integer('num_layers', 2, 'Number of layers in VAE (if conv=False)')
    cl.DEFINE_float('noise_pixel_var', 0.1, 'Noise variance for the pixels in the image if out_distr=gaussian')
    cl.DEFINE_float('ll_keep_prob', 1.0, 'Keep probability of p(x|a)')
    cl.DEFINE_boolean('use_vae', True, 'Use VAE and not AE')

    # LGSSM config
    cl.DEFINE_integer('dim_a', 2, 'Number of latent variables in the VAE')
    cl.DEFINE_integer('dim_z', 4, 'Dimension of the latent state in the LGSSM')
    cl.DEFINE_integer('dim_u', 1, 'Dimension of the inputs')
    cl.DEFINE_integer('K', 3, 'Number of filters in mixture')
    cl.DEFINE_float('noise_emission', 0.03, 'Noise level for the measurement noise matrix')
    cl.DEFINE_float('noise_transition', 0.08, 'Noise level for the process noise matrix')
    cl.DEFINE_float('init_cov', 20.0, 'Variance of the initial state')

    # Parameter network config
    cl.DEFINE_boolean('alpha_rnn', True, 'Use LSTM RNN for alpha')
    cl.DEFINE_integer('alpha_units', 50, 'Number of units in alpha network')
    cl.DEFINE_integer('alpha_layers', 2, 'Number of layers in alpha network (if alpha_rnn=False)')
    cl.DEFINE_string('alpha_activation', 'relu', 'Activation function in alpha (if alpha_rnn=False)')
    cl.DEFINE_integer('fifo_size', 1, 'Number of items in the alpha FIFO memory (if alpha_rnn=False)')
    cl.DEFINE_boolean('learn_u', False, 'Model u from a neural network')

    # Training config
    cl.DEFINE_integer('batch_size', 32, 'Size of mini batches')
    cl.DEFINE_float('init_lr', 0.007, 'Starter learning rate')
    cl.DEFINE_float('init_kf_matrices', 0.05, 'Standard deviation of noise used to initialize B and C')
    cl.DEFINE_float('max_grad_norm', 150.0, 'Gradient norm clipping')
    cl.DEFINE_float('scale_reconstruction', 0.3, 'Scale for the reconstruction term in the elbo')
    cl.DEFINE_integer('only_vae_epochs', 0, 'Number of epochs in which we only train the vae')
    cl.DEFINE_integer('kf_update_steps', 10, 'Number of extra update steps done for the kalman filter')
    cl.DEFINE_float('decay_rate', 0.85, 'Decay steps for exponential lr decay')
    cl.DEFINE_integer('decay_steps', 20, 'Decay steps for exponential lr decay')
    cl.DEFINE_boolean('sample_z', False, 'Sample z from the prior')
    cl.DEFINE_integer("num_epochs", 80, "Epoch to train")
    cl.DEFINE_float('train_miss_prob', 0.0, 'Fraction of missing data during training')
    cl.DEFINE_integer('t_init_train_miss', 3, 'Number of initial observed frames when training with missing data')

    # Utils config
    cl.DEFINE_string('gpu', '', 'Comma seperated list of GPUs')
    cl.DEFINE_string('reload_model', '', 'Path to the model.cpkt file')

    # Logs/plotting config
    cl.DEFINE_string('run_name', time.strftime('%Y%m%d%H%M%S', time.localtime()), 'Name for the run')
    cl.DEFINE_string('log_dir', 'logs', 'Directory to save files in')
    cl.DEFINE_integer('display_step', 1, 'After how many epochs to print logs')
    cl.DEFINE_integer('generate_step', 20, 'After how many epochs to store the plots')
    cl.DEFINE_integer('n_steps_gen', 80, 'Number of steps to generate in a sequence')
    cl.DEFINE_integer('t_init_mask', 4, 'Initial time step for data imputation')
    cl.DEFINE_integer('t_steps_mask', 12, 'Number of time steps for data imputation')

    return cl


if __name__ == '__main__':
    config = get_image_config()
    config.DEFINE_bool('test', True, 'test')
    config = reload_config(config.FLAGS)

    print(config.dataset)
    config.dataset = 'test'
    print(config.dataset)

    print(config.__flags)
