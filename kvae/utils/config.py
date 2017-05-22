import tensorflow as tf
import time
import os
import json


def reload_config(FLAGS):
    # If we are reloading a model, overwrite the flags
    # TODO: We could only overwrite the model related flags, if we want e.g. to start training with different params
    if FLAGS.reload_model is not '':
        with open('%s/%s' % (os.path.dirname(FLAGS.reload_model), 'config.json')) as data_file:
            config_dict = json.load(data_file)

        for key, value in config_dict.items():
            attr_remove = ['gpu', 'run_name', 'log_dir', 'n_steps_gen', 'reload_model', 'display_step', 'generate_step']
            # attr_remove = ['gpu', 'n_steps_gen', 'reload_model', 'display_step', 'generate_step']
            if key not in attr_remove:
                FLAGS.__setattr__(key, value)
    return FLAGS


def base_config():
    cl = tf.app.flags

    cl.DEFINE_string('activation', 'relu', 'Activation function in VAE')
    cl.DEFINE_string('alpha_activation', 'relu', 'Activation function in alpha')
    cl.DEFINE_integer('alpha_layers', 2, 'Number of layers in alpha network')
    cl.DEFINE_boolean('alpha_gumbel', False, 'Use gumbel softmax distribution on alpha')
    cl.DEFINE_integer('alpha_units', 50, 'Number of units in alpha network')
    cl.DEFINE_boolean('alpha_rnn', True, 'Use GRU RNN for alpha')
    cl.DEFINE_string('dataset', 'box_rnd', 'Select data set')
    cl.DEFINE_integer('display_step', 1, 'After how many epochs to print')
    cl.DEFINE_integer('fifo_size', 1, 'Number of items in the alpha FIFO memory')
    cl.DEFINE_integer('generate_step', 20, 'After how many epochs to print')
    cl.DEFINE_string('gpu', '', 'Comma seperated list of GPUs')
    cl.DEFINE_integer('gumbel_decay_steps', 10, 'Steps to decay gumbel tau temperature')
    cl.DEFINE_float('gumbel_min', 0.5, 'Minimum gumbel tau temperature')
    cl.DEFINE_boolean('learn_u', False, 'Model u from a neural network')
    cl.DEFINE_string('log_dir', 'logs', 'Directory to save files in')
    cl.DEFINE_float('ll_keep_prob', 1.0, 'Keep probability of p(x|a)')
    cl.DEFINE_float('max_grad_norm', 150.0, 'Gradient norm clipping')
    cl.DEFINE_integer('n_steps_gen', 80, 'Number of steps to generate')
    cl.DEFINE_string('out_distr', 'bernoulli', 'Output distibution')
    cl.DEFINE_string('reload_model', '', 'Path to the model.cpkt file')
    cl.DEFINE_string('run_name', time.strftime('%Y%m%d%H%M%S', time.localtime()), 'Name for the run')
    cl.DEFINE_float('scale_reconstruction', 0.3, 'Scale for the reconstruction term in the elbo')
    cl.DEFINE_boolean('temp_entropy', False, 'Use temperature only on the entropy term')
    cl.DEFINE_integer('temp_steps', 25, 'Temperature steps for the annealing of the kalman filter elbo')
    cl.DEFINE_integer('t_init_mask', 4, 'Initial time step for data imputation')
    cl.DEFINE_integer('t_steps_mask', 12, 'Number of time steps for data imputation')
    cl.DEFINE_float('train_miss_prob', 0.0, 'Fraction of missing data during training')
    cl.DEFINE_integer('t_init_train_miss', 3, 'Number of missing frames during training')
    return cl


def get_2d_config():
    cl = base_config()
    cl.DEFINE_integer('batch_size', 100, 'Size of mini batches')
    cl.DEFINE_float('decay_rate', 0.8, 'Decay steps for exponential lr decay')
    cl.DEFINE_integer('decay_steps', 30, 'Decay steps for exponential lr decay')
    cl.DEFINE_integer('dim_u', 1, 'Dimension of the inputs')
    cl.DEFINE_integer('dim_z', 5, 'Dimension of the latent state')
    cl.DEFINE_integer('dim_a', 2, 'Number of latent variables in the VAE')
    cl.DEFINE_integer('K', 4, 'Number of filters in mixture')
    cl.DEFINE_float('init_cov', 20.0, 'Variance of the initial state')
    cl.DEFINE_float('init_kf_matrices', 0.05, 'Standard deviation of noise used to initialize B and C')
    cl.DEFINE_float('init_lr', 0.008, 'Starter learning rate')
    cl.DEFINE_float('init_temp', 0.1, 'Starter temperature for the annealing of the kalman filter elbo')
    cl.DEFINE_integer('kf_update_steps', 20, 'Number of extra update steps done for the kalman filter')
    cl.DEFINE_float('noise_emission', 0.01, 'Noise level for the measurement noise matrix')
    cl.DEFINE_float('noise_pixel_var', 0.1, 'Noise variance for the pixels in the image')
    cl.DEFINE_float('noise_transition', 0.1, 'Noise level for the process noise matrix')
    cl.DEFINE_integer("num_epochs", 200, "Epoch to train")
    return cl


def get_image_config():
    cl = base_config()
    cl.DEFINE_integer('batch_size', 32, 'Size of mini batches')
    cl.DEFINE_boolean('conv', True, 'Use conv vae')
    cl.DEFINE_float('decay_rate', 0.85, 'Decay steps for exponential lr decay')
    cl.DEFINE_integer('decay_steps', 20, 'Decay steps for exponential lr decay')
    cl.DEFINE_integer('dim_u', 1, 'Dimension of the inputs')
    cl.DEFINE_integer('dim_z', 4, 'Dimension of the latent state')
    cl.DEFINE_integer('dim_a', 2, 'Number of latent variables in the VAE')
    cl.DEFINE_integer('filter_size', 3, 'Filter size in conv vae')
    cl.DEFINE_float('init_cov', 20.0, 'Variance of the initial state')
    cl.DEFINE_float('init_kf_matrices', 0.05, 'Standard deviation of noise used to initialize B and C')
    cl.DEFINE_float('init_lr', 0.007, 'Starter learning rate')
    cl.DEFINE_float('init_temp', 1.0, 'Starter temperature for the annealing of the kalman filter elbo')
    cl.DEFINE_integer('K', 3, 'Number of filters in mixture')
    cl.DEFINE_integer('kf_update_steps', 10, 'Number of extra update steps done for the kalman filter')
    cl.DEFINE_float('noise_emission', 0.03, 'Noise level for the measurement noise matrix')
    cl.DEFINE_float('noise_pixel_var', 0.1, 'Noise variance for the pixels in the image')
    cl.DEFINE_float('noise_transition', 0.08, 'Noise level for the process noise matrix')
    cl.DEFINE_string('num_filters', '32,32,32', 'Comma separated list of conv filters')
    cl.DEFINE_integer('num_layers', 2, 'Number of layers in VAE')
    cl.DEFINE_integer('only_vae_epochs', 0, 'Number of epochs in which we only train the vae')
    cl.DEFINE_boolean('sample_z', False, 'Sample z from the prior')
    cl.DEFINE_boolean('use_vae', True, 'Use VAE and not AE')
    cl.DEFINE_integer('vae_num_units', 25, 'Number of hidden units in the VAE')
    cl.DEFINE_integer("num_epochs", 80, "Epoch to train")
    return cl


def get_rnn_config():
    cl = get_image_config()
    cl.DEFINE_integer('rnn_units', 64, 'Units in RNN cell')
    return cl


if __name__ == '__main__':
    # config = Config()
    config = get_2d_config()
    config.DEFINE_bool('test', True, 'test')
    config = reload_config(config.FLAGS)

    print(config.dataset)
    config.dataset = 'test'
    # config.update('dataset', 'test')
    print(config.dataset)

    print(config.__flags)
