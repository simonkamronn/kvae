from tensorflow.contrib import slim
from tensorflow.contrib.layers import optimize_loss
from kvae.filter import KalmanFilter
from kvae.utils.movie import save_frames, save_movies_to_frame, save_true_generated_frames, save_movie_to_frame
from kvae.utils.plotting import (plot_auxiliary, plot_alpha_grid, plot_ball_trajectories, plot_trajectory_and_video,
                                 plot_3d_ball_trajectory, plot_ball_trajectories_comparison,
                                 plot_ball_and_alpha, plot_ball_trajectory, plot_alpha)
from kvae.utils.data import PymunkData
from kvae.utils.nn import *
from tensorflow.contrib.rnn import BasicLSTMCell

import time
from scipy.spatial.distance import hamming
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})

np.random.seed(1337)


class KalmanVariationalAutoencoder(object):
    """ This class defines functions to build, train and evaluate Kalman Variational Autoencoders
    """
    def __init__(self, config, sess):
        self.config = config

        # Load the dataset
        self.train_data = PymunkData("../data/{}.npz".format(config.dataset), config)
        self.test_data = PymunkData("../data/{}_test.npz".format(config.dataset), config)

        # Frame sizes
        self.d1 = self.train_data.d1
        self.d2 = self.train_data.d2

        # Initializers for LGSSM variables. A is intialized with identity matrices, B and C randomly from a gaussian
        A = np.array([np.eye(config.dim_z).astype(np.float32) for _ in range(config.K)])
        B = np.array([config.init_kf_matrices * np.random.randn(config.dim_z, config.dim_u).astype(np.float32)
                      for _ in range(config.K)])
        C = np.array([config.init_kf_matrices * np.random.randn(config.dim_a, config.dim_z).astype(np.float32)
                      for _ in range(config.K)])
        # We use isotropic covariance matrices
        Q = config.noise_transition * np.eye(config.dim_z, dtype=np.float32)
        R = config.noise_emission * np.eye(config.dim_a, dtype=np.float32)

        # p(z_1)
        mu = np.zeros((self.config.batch_size, config.dim_z), dtype=np.float32)
        Sigma = np.tile(config.init_cov * np.eye(config.dim_z, dtype=np.float32), (self.config.batch_size, 1, 1))

        # Initial variable a_0
        a_0 = np.zeros((config.dim_a,), dtype=np.float32)

        # Collect initial variables
        self.init_vars = dict(A=A, B=B, C=C, Q=Q, R=R, mu=mu, Sigma=Sigma, a_0=a_0)

        # Get activation function for hidden layers
        if config.activation.lower() == 'relu':
            self.activation_fn = tf.nn.relu
        elif config.activation.lower() == 'tanh':
            self.activation_fn = tf.nn.tanh
        elif config.activation.lower() == 'elu':
            self.activation_fn = tf.nn.elu
        else:
            self.activation_fn = None

        # Parse num_filters to list of ints
        self.num_filters = [int(f) for f in config.num_filters.split(',')]

        # Set Tensorflow session
        self.sess = sess

        # Init placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, None, self.d1, self.d2], name='x')
        self.ph_steps = tf.placeholder(tf.int32, shape=(), name='n_step')
        self.scale_reconstruction = tf.placeholder(tf.float32, shape=(), name='scale_reconstruction')
        self.mask = tf.placeholder(tf.float32, shape=(None, None), name='mask')
        self.a_prev = tf.placeholder(tf.float32, shape=[None, config.dim_a], name='a_prev')  # For alpha NN plotting

        # Init various
        self.saver = None
        self.kf = None
        self.vae_updates = None
        self.vae_kf_updates = None
        self.all_updates = None
        self.lb_vars = None
        self.model_vars = None
        self.enc_shape = None
        self.n_steps_gen = None
        self.out_gen_det = None
        self.out_gen = None
        self.out_gen_det_impute = None
        self.train_summary = None
        self.test_summary = None

    def encoder(self, x):
        """ Convolutional variational encoder to encode image into a low-dimensional latent code
        If config.conv == False it is a MLP VAE. If config.use_vae == False, it is a normal encoder
        :param x: sequence of images
        :return: a, a_mu, a_var
        """
        with tf.variable_scope('vae/encoder'):
            if self.config.conv:
                x_flat_conv = tf.reshape(x, (-1, self.d1, self.d2, 1))
                enc_hidden = slim.stack(x_flat_conv,
                                        slim.conv2d,
                                        self.num_filters,
                                        kernel_size=self.config.filter_size,
                                        stride=2,
                                        activation_fn=self.activation_fn,
                                        padding='SAME')
                enc_flat = slim.flatten(enc_hidden)
                self.enc_shape = enc_hidden.get_shape().as_list()[1:]

            else:
                x_flat = tf.reshape(x, (-1, self.d1 * self.d2))
                enc_flat = slim.repeat(x_flat, self.config.num_layers, slim.fully_connected,
                                       self.config.vae_num_units, self.activation_fn)

            a_mu = slim.fully_connected(enc_flat, self.config.dim_a, activation_fn=None)
            if self.config.use_vae:
                a_var = slim.fully_connected(enc_flat, self.config.dim_a, activation_fn=tf.nn.sigmoid)
                a_var = self.config.noise_emission * a_var
                a = simple_sample(a_mu, a_var)
            else:
                a_var = tf.constant(1., dtype=tf.float32, shape=())
                a = a_mu
            a_seq = tf.reshape(a, tf.stack((-1, self.ph_steps, self.config.dim_a)))
        return a_seq, a_mu, a_var

    def decoder(self, a_seq):
        """ Convolutional variational decoder to decode latent code to image reconstruction
        If config.conv == False it is a MLP VAE. If config.use_vae == False it is a normal decoder
        :param a_seq: latent code
        :return: x_hat, x_mu, x_var
        """
        # Create decoder
        if self.config.out_distr == 'bernoulli':
            activation_x_mu = tf.nn.sigmoid
        else:
            activation_x_mu = None

        with tf.variable_scope('vae/decoder'):
            a = tf.reshape(a_seq, (-1, self.config.dim_a))
            if self.config.conv:
                dec_upscale = slim.fully_connected(a, int(np.prod(self.enc_shape)), activation_fn=None)
                dec_upscale = tf.reshape(dec_upscale, [-1] + self.enc_shape)

                dec_hidden = dec_upscale
                for filters in reversed(self.num_filters):
                    dec_hidden = slim.conv2d(dec_hidden, filters * 4, self.config.filter_size,
                                             activation_fn=self.activation_fn)
                    dec_hidden = subpixel_reshape(dec_hidden, 2)
                x_mu = slim.conv2d(dec_hidden, 1, 1, stride=1, activation_fn=activation_x_mu)
                x_var = tf.constant(self.config.noise_pixel_var, dtype=tf.float32, shape=())
            else:
                dec_hidden = slim.repeat(a, self.config.num_layers, slim.fully_connected,
                                         self.config.vae_num_units, self.activation_fn)

                x_mu = slim.fully_connected(dec_hidden, self.d1 * self.d2, activation_fn=activation_x_mu)
                x_mu = tf.reshape(x_mu, (-1, self.d1, self.d2, 1))
                # x_var is not used for bernoulli outputs. Here we fix the output variance of the Gaussian,
                # we could also learn it globally for each pixel (as we did in the pendulum experiment) or through a
                # neural network.
                x_var = tf.constant(self.config.noise_pixel_var, dtype=tf.float32, shape=())

        if self.config.out_distr == 'bernoulli':
            # For bernoulli we show the probabilities
            x_hat = x_mu
        else:
            x_hat = simple_sample(x_mu, x_var)

        return tf.reshape(x_hat, tf.stack((-1, self.ph_steps, self.d1, self.d2))), x_mu, x_var

    def alpha(self, inputs, state=None, u=None, buffer=None, reuse=None, init_buffer=False, name='alpha'):
        """The dynamics parameter network alpha for mixing transitions in a state space model.
        This function is quite general and supports different architectures (NN, RNN, FIFO queue, learning the inputs)

        Args:
            inputs: tensor to condition mixing vector on
            state: previous state if using RNN network to model alpha
            u: pass-through variable if u is given (learn_u=False)
            buffer: buffer for the FIFO network (used for fifo_size>1)
            reuse: `True` or `None`; if `True`, we go into reuse mode for this scope as
                    well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
            init_buffer: initialize buffer for a_t
            name: name of the scope

        Returns:
            alpha: mixing vector of dimension (batch size, K)
            state: new state
            u: either inferred u from model or pass-through
            buffer: FIFO buffer
        """
        # Increase the number of hidden units if we also learn u (learn_u=True)
        num_units = self.config.alpha_units * 2 if self.config.learn_u else self.config.alpha_units

        # Overwrite input buffer
        if init_buffer:
            buffer = tf.zeros((tf.shape(inputs)[0], self.config.dim_a, self.config.fifo_size), dtype=tf.float32)

        # If K == 1, return inputs
        if self.config.K == 1:
            return tf.ones([self.config.batch_size, self.config.K]), state, u, buffer

        with tf.variable_scope(name, reuse=reuse):
            if self.config.alpha_rnn:
                rnn_cell = BasicLSTMCell(num_units, reuse=reuse)
                output, state = rnn_cell(inputs, state)
            else:
                # Shift buffer
                buffer = tf.concat([buffer[:, :, 1:], tf.expand_dims(inputs, 2)], 2)
                output = slim.repeat(
                    tf.reshape(buffer, (tf.shape(inputs)[0], self.config.dim_a * self.config.fifo_size)),
                    self.config.alpha_layers, slim.fully_connected, num_units,
                    get_activation_fn(self.config.alpha_activation), scope='hidden')

            # Get Alpha as the first part of the output
            alpha = slim.fully_connected(output[:, :self.config.alpha_units],
                                         self.config.K,
                                         activation_fn=tf.nn.softmax,
                                         scope='alpha_var')

            if self.config.learn_u:
                # Get U as the second half of the output
                u = slim.fully_connected(output[:, self.config.alpha_units:],
                                         self.config.dim_u, activation_fn=None, scope='u_var')
        return alpha, state, u, buffer

    def build_model(self):
        # Encoder q(a|x)
        a_seq, a_mu, a_var = self.encoder(self.x)
        a_vae = a_seq

        # Initial state for the alpha RNN
        dummy_lstm = BasicLSTMCell(self.config.alpha_units * 2 if self.config.learn_u else self.config.alpha_units)
        state_init_rnn = dummy_lstm.zero_state(self.config.batch_size, tf.float32)

        # Initialize Kalman filter (LGSSM)
        self.kf = KalmanFilter(dim_z=self.config.dim_z,
                               dim_y=self.config.dim_a,
                               dim_u=self.config.dim_u,
                               dim_k=self.config.K,
                               A=self.init_vars['A'],  # state transition function
                               B=self.init_vars['B'],  # control matrix
                               C=self.init_vars['C'],  # Measurement function
                               R=self.init_vars['R'],  # measurement noise
                               Q=self.init_vars['Q'],  # process noise
                               y=a_seq,  # output
                               u=None,
                               mask=self.mask,
                               mu=self.init_vars['mu'],
                               Sigma=self.init_vars['Sigma'],
                               y_0=self.init_vars['a_0'],
                               alpha=self.alpha,
                               state=state_init_rnn
                               )

        # Get smoothed posterior over z
        smooth, A, B, C, alpha_plot = self.kf.smooth()

        # Get filtered posterior, used only for imputation plots
        filter, _, _, C_filter, _ = self.kf.filter()

        # Get a from the prior z (for plotting)
        a_mu_pred = tf.matmul(C, tf.expand_dims(smooth[0], 2), transpose_b=True)
        a_mu_pred_seq = tf.reshape(a_mu_pred, tf.stack((-1, self.ph_steps, self.config.dim_a)))
        if self.config.sample_z:
            a_seq = a_mu_pred_seq

        # Decoder p(x|a)
        x_hat, x_mu, x_var = self.decoder(a_seq)

        # Compute variables for generation from the model (for plotting)
        self.n_steps_gen = self.config.n_steps_gen  # We sample for this many iterations,
        self.out_gen_det = self.kf.sample_generative_tf(smooth, self.n_steps_gen, deterministic=True,
                                                        init_fixed_steps=self.config.t_init_mask)
        self.out_gen = self.kf.sample_generative_tf(smooth, self.n_steps_gen, deterministic=False,
                                                    init_fixed_steps=self.config.t_init_mask)
        self.out_gen_det_impute = self.kf.sample_generative_tf(smooth, self.test_data.timesteps, deterministic=True,
                                                               init_fixed_steps=self.config.t_init_mask)
        self.out_alpha, _, _, _ = self.alpha(self.a_prev, state=state_init_rnn, u=None, init_buffer=True, reuse=True)

        # Collect generated model variables
        self.model_vars = dict(x_hat=x_hat, x_mu=x_mu, x_var=x_var,
                               a_seq=a_seq, a_mu=a_mu, a_var=a_var, a_vae=a_vae,
                               smooth=smooth, A=A, B=B, C=C, alpha_plot=alpha_plot,
                               a_mu_pred_seq=a_mu_pred_seq, filter=filter, C_filter=C_filter)

        return self

    def build_loss(self):
        # Reshape x for log_likelihood
        x_flat = tf.reshape(self.x, (-1, self.d1 * self.d2))
        x_mu_flat = tf.reshape(self.model_vars['x_mu'], (-1, self.d1 * self.d2))
        mask_flat = tf.reshape(self.mask, (-1,))

        # VAE loss
        elbo_vae, log_px, log_qa = log_likelihood(x_mu_flat,
                                                  self.model_vars['x_var'],
                                                  x_flat,
                                                  self.model_vars['a_mu'],
                                                  self.model_vars['a_var'],
                                                  tf.reshape(self.model_vars['a_vae'], (-1, self.config.dim_a)),
                                                  mask_flat,
                                                  self.config)
        # LGSSM loss
        elbo_kf, kf_log_probs, z_smooth = self.kf.get_elbo(self.model_vars['smooth'],
                                                           self.model_vars['A'],
                                                           self.model_vars['B'],
                                                           self.model_vars['C'])

        # Calc number of batches
        num_batches = self.train_data.sequences // self.config.batch_size

        # Decreasing learning rate
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(self.config.init_lr, global_step,
                                                   self.config.decay_steps * num_batches,
                                                   self.config.decay_rate, staircase=True)

        # Combine individual ELBO's
        elbo_tot = self.scale_reconstruction * log_px + elbo_kf - log_qa

        # Collect variables to monitor lb
        self.lb_vars = [elbo_tot, elbo_kf, kf_log_probs, elbo_vae, log_px, log_qa]

        # Get list of vars for gradient computation
        vae_vars = slim.get_variables('vae')
        kf_vars = [self.kf.A, self.kf.B, self.kf.C, self.kf.y_0]
        all_vars = tf.trainable_variables()

        # Define training updates
        self.vae_updates = optimize_loss(loss=-elbo_tot,
                                         global_step=global_step,
                                         learning_rate=learning_rate,
                                         optimizer='Adam',
                                         clip_gradients=self.config.max_grad_norm,
                                         variables=vae_vars,
                                         summaries=["gradients", "gradient_norm"],
                                         name='vae_updates')

        self.vae_kf_updates = optimize_loss(loss=-elbo_tot,
                                        global_step=global_step,
                                        learning_rate=learning_rate,
                                        optimizer='Adam',
                                        clip_gradients=self.config.max_grad_norm,
                                        variables=kf_vars + vae_vars,
                                        summaries=["gradients", "gradient_norm"],
                                        name='vae_kf_updates')

        self.all_updates = optimize_loss(loss=-elbo_tot,
                                         global_step=global_step,
                                         learning_rate=learning_rate,
                                         optimizer='Adam',
                                         clip_gradients=self.config.max_grad_norm,
                                         variables=all_vars,
                                         summaries=["gradients", "gradient_norm"],
                                         name='all_updates')

        tf.summary.scalar('learningrate', learning_rate)
        tf.summary.scalar('mean_var_qa', tf.reduce_mean(self.model_vars['a_var']))
        return self

    def initialize_variables(self):
        """ Initialize variables or load saved model
        :return: self
        """
        # Setup saver
        self.saver = tf.train.Saver()

        # Initialize or reload variables
        if self.config.reload_model is not '':
            print("Restoring model in %s" % self.config.reload_model)
            self.saver.restore(self.sess, self.config.reload_model)
        else:
            self.sess.run(tf.global_variables_initializer())
        return self

    def train(self):
        """ Train model given parameters in self.config
        :return: imputation error on test set
        """
        sess = self.sess
        writer = tf.summary.FileWriter(self.config.log_dir, sess.graph)
        num_batches = self.train_data.sequences // self.config.batch_size
        # This code supports training with missing data (if train_miss_prob > 0.0)
        mask_train = np.ones((num_batches, self.config.batch_size, self.train_data.timesteps), dtype=np.float32)
        if self.config.train_miss_prob > 0.0:
            # Always use the same mask for each sequence during training
            for j in range(num_batches):
                mask_train[j] = self.mask_impute_random(t_init_mask=self.config.t_init_train_miss,
                                                        drop_prob=self.config.train_miss_prob)
        all_summaries = tf.summary.merge_all()

        for n in range(self.config.num_epochs):
            elbo_tot = []
            elbo_kf = []
            kf_log_probs = []
            elbo_vae = []
            log_px = []
            log_qa = []
            time_epoch_start = time.time()
            for i in range(num_batches):
                slc = slice(i * self.config.batch_size, (i + 1) * self.config.batch_size)
                feed_dict = {self.x: self.train_data.images[slc],
                             self.kf.u: self.train_data.controls[slc],
                             self.mask: mask_train[i],
                             self.ph_steps: self.train_data.timesteps,
                             self.scale_reconstruction: self.config.scale_reconstruction}

                # Support for different updates schemes. It is beneficial to achieve better convergence not to train
                # alpha from the beginning
                if n < self.config.only_vae_epochs:
                    sess.run(self.vae_updates, feed_dict)
                elif n < self.config.only_vae_epochs + self.config.kf_update_steps:
                    sess.run(self.vae_kf_updates, feed_dict)
                else:
                    sess.run(self.all_updates, feed_dict)

                # Bookkeeping.
                _elbo_tot, _elbo_kf, _kf_log_probs, _elbo_vae, _log_px, _log_qa = sess.run(self.lb_vars, feed_dict)
                elbo_tot.append(_elbo_tot)
                elbo_kf.append(_elbo_kf)
                kf_log_probs.append(_kf_log_probs)
                elbo_vae.append(_elbo_vae)
                log_px.append(_log_px)
                log_qa.append(_log_qa)

            # Write to summary
            summary_train = self.def_summary('train', elbo_tot, elbo_kf, kf_log_probs, elbo_vae, log_px, log_qa)
            writer.add_summary(summary_train, n)
            writer.add_summary(sess.run(all_summaries, feed_dict), n)

            if (n + 1) % self.config.display_step == 0:
                mean_kf_log_probs = np.mean(kf_log_probs, axis=0)
                print("Epoch %d, ELBO %.2f, log_probs [%.2f, %.2f, %.2f, %.2f], elbo_vae %.2f, took %.2fs"
                      % (n, np.mean(elbo_tot), mean_kf_log_probs[0], mean_kf_log_probs[1],
                         mean_kf_log_probs[2], mean_kf_log_probs[3], np.mean(elbo_vae),
                         time.time() - time_epoch_start))

            if (((n + 1) % self.config.generate_step == 0) and n > 0) or (n == self.config.num_epochs - 1) or (n == 0):
                # Impute and calculate error
                mask_impute = self.mask_impute_planning(t_init_mask=self.config.t_init_mask,
                                                        t_steps_mask=self.config.t_steps_mask)
                out_res = self.impute(mask_impute, t_init_mask=self.config.t_init_mask, n=n)

                # Generate sequences for evaluation
                self.generate(n=n)

                # Test on previously unseen data
                test_elbo, summary_test = self.test()
                writer.add_summary(summary_test, n)

        # Save the last model
        self.saver.save(sess, self.config.log_dir + '/model.ckpt')
        neg_lower_bound = -np.mean(test_elbo)
        print("Negative lower_bound on the test set: %s" % neg_lower_bound)
        return out_res[0]

    def test(self):
        mask_test = np.ones((self.config.batch_size, self.test_data.timesteps), dtype=np.float32)

        elbo_tot = []
        elbo_kf = []
        kf_log_probs = []
        elbo_vae = []
        log_px = []
        log_qa = []
        time_test_start = time.time()
        for i in range(self.test_data.sequences // self.config.batch_size):
            slc = slice(i * self.config.batch_size, (i + 1) * self.config.batch_size)
            feed_dict = {self.x: self.test_data.images[slc],
                         self.kf.u: self.test_data.controls[slc],
                         self.mask: mask_test,
                         self.ph_steps: self.test_data.timesteps,
                         self.scale_reconstruction: 1.0}

            # Bookkeeping.
            _elbo_tot, _elbo_kf, _kf_log_probs, _elbo_vae, _log_px, _log_qa  = self.sess.run(self.lb_vars, feed_dict)
            elbo_tot.append(_elbo_tot)
            elbo_kf.append(_elbo_kf)
            kf_log_probs.append(_kf_log_probs)
            elbo_vae.append(_elbo_vae)
            log_px.append(_log_px)
            log_qa.append(_log_qa)

        # Write to summary
        summary = self.def_summary('test', elbo_tot, elbo_kf, kf_log_probs, elbo_vae, log_px, log_qa)
        mean_kf_log_probs = np.mean(kf_log_probs, axis=0)
        print("-- TEST, ELBO %.2f, log_probs [%.2f, %.2f, %.2f, %.2f], elbo_vae %.2f, took %.2fs"
              % (np.mean(elbo_tot), mean_kf_log_probs[0], mean_kf_log_probs[1],
                 mean_kf_log_probs[2], mean_kf_log_probs[3], np.mean(elbo_vae),
                 time.time() - time_test_start))
        return np.mean(elbo_tot), summary

    def generate(self, idx_batch=0, n=99999):
        ###### Sample video deterministic ######
        # Get initial state z_1
        mask_test = np.ones((self.config.batch_size, self.test_data.timesteps), dtype=np.float32)
        slc = slice(idx_batch * self.config.batch_size, (idx_batch + 1) * self.config.batch_size)
        feed_dict = {self.x: self.test_data.images[slc],
                     self.kf.u: self.test_data.controls[slc],
                     self.ph_steps: self.test_data.timesteps,
                     self.mask: mask_test}
        smooth_z = self.sess.run(self.model_vars['smooth'], feed_dict)

        # Sample deterministic generation
        feed_dict = {self.model_vars['smooth']: smooth_z,
                     self.kf.u: np.zeros((self.config.batch_size, self.n_steps_gen, self.config.dim_u)),
                     self.ph_steps: self.n_steps_gen}
        a_gen_det, _, alpha_gen_det = self.sess.run(self.out_gen_det, feed_dict)
        x_gen_det = self.sess.run(self.model_vars['x_hat'], {self.model_vars['a_seq']: a_gen_det,
                                                             self.ph_steps: self.n_steps_gen})

        # Save the trajectory of deterministic a (we only plot the first 2 dimensions!) and alpha
        plot_ball_trajectories(a_gen_det, self.config.log_dir + '/plot_generation_det_%05d.png' % n)
        plot_alpha_grid(alpha_gen_det, self.config.log_dir + '/alpha_generation_det_%05d.png' % n)

        # Sample stochastic
        a_gen, _, alpha_gen = self.sess.run(self.out_gen, feed_dict)
        x_gen = self.sess.run(self.model_vars['x_hat'], {self.model_vars['a_seq']: a_gen,
                                                         self.ph_steps: self.n_steps_gen})

        # Save movies
        save_frames(x_gen, self.config.log_dir + '/video_generation_%05d.mp4' % n)
        save_frames(x_gen_det, self.config.log_dir + '/video_generation_det_%05d.mp4' % n)

        # Save stochastic a and alpha
        plot_ball_trajectories(a_gen, self.config.log_dir + '/plot_generation_%05d.png' % n)
        plot_alpha_grid(alpha_gen, self.config.log_dir + '/alpha_generation_%05d.png' % n)

        # Save movie to single frame
        save_movies_to_frame(x_gen_det[:, :20], self.config.log_dir + '/video_generation_image_%05d.png' % n)

        # We can only show the image for alpha when using a simple neural network
        if self.config.dim_a == 2 and self.config.fifo_size == 1 and self.config.alpha_rnn == False \
                and self.config.learn_u == False:
            self.img_alpha_nn(n=n, range_x=(-16, 16), range_y=(-16, 16))

    def impute(self, mask_impute, t_init_mask, idx_batch=0, n=99999, plot=True):
        slc = slice(idx_batch * self.config.batch_size, (idx_batch + 1) * self.config.batch_size)
        feed_dict = {self.x: self.test_data.images[slc],
                     self.kf.u: self.test_data.controls[slc],
                     self.ph_steps: self.test_data.timesteps,
                     self.mask: mask_impute}

        ##### Compute reconstructions and imputations (smoothing) ######
        a_imputed, a_reconstr, x_reconstr, alpha_reconstr, smooth_z, filter_z, C_filter = self.sess.run([
                                                                                     self.model_vars['a_mu_pred_seq'],
                                                                                     self.model_vars['a_vae'],
                                                                                     self.model_vars['x_hat'],
                                                                                     self.model_vars['alpha_plot'],
                                                                                     self.model_vars['smooth'],
                                                                                     self.model_vars['filter'],
                                                                                     self.model_vars['C_filter']],
                                                                                    feed_dict)
        x_imputed = self.sess.run(self.model_vars['x_hat'], {self.model_vars['a_seq']: a_imputed,
                                                             self.ph_steps: self.test_data.timesteps})
        x_true = feed_dict[self.x]

        ###### Filtering
        feed_dict = {self.model_vars['smooth']: filter_z,
                     self.model_vars['C']: C_filter,
                     self.ph_steps: self.test_data.timesteps}
        a_filtered = self.sess.run(self.model_vars['a_mu_pred_seq'], feed_dict)
        x_filtered = self.sess.run(self.model_vars['x_hat'], {self.model_vars['a_seq']: a_filtered,
                                                              self.ph_steps: self.test_data.timesteps})
        if plot:
            save_frames(x_true, self.config.log_dir + '/video_true.mp4')
            save_frames(x_imputed, self.config.log_dir + '/video_smoothing_%05d.mp4' % n)
            save_frames(x_reconstr, self.config.log_dir + '/video_reconstruction_%05d.mp4' % n)
            save_true_generated_frames(x_true, x_imputed, self.config.log_dir + '/video_true_smooth_%05d.mp4' % n)
            save_true_generated_frames(x_true, x_filtered, self.config.log_dir + '/video_true_filter_%05d.mp4' % n)
            save_true_generated_frames(x_true, x_reconstr, self.config.log_dir + '/video_true_recon_%05d.mp4' % n)

            plot_alpha_grid(alpha_reconstr, self.config.log_dir + '/alpha_reconstr_%05d.png' % n)

            # Plot z_mu
            plot_auxiliary([smooth_z[0]], self.config.log_dir + '/plot_z_mu_smooth_%05d.png' % n)

            if self.config.dim_a == 2:
                # Plot alpha and corresponding trajectory
                plot_ball_and_alpha(alpha_reconstr[14], a_reconstr[14], cmap='Reds',
                                    filename=self.config.log_dir + '/alpha_a_recon_%05d.png' % n)

        ###### Sample deterministic generation having access to the first t_init_mask frames for comparison
        # Get initial state z_1
        feed_dict = {self.x: self.test_data.images[slc][:, 0: t_init_mask],
                     self.kf.u: self.test_data.controls[slc][:, 0: t_init_mask],
                     self.ph_steps: t_init_mask,
                     self.mask: mask_impute[:, 0: t_init_mask]}
        smooth_z_gen = self.sess.run(self.model_vars['smooth'], feed_dict)
        feed_dict = {self.model_vars['smooth']: smooth_z_gen,
                     self.kf.u: np.zeros((self.config.batch_size, self.n_steps_gen, self.config.dim_u)),
                     self.ph_steps: self.test_data.timesteps}
        a_gen_det, _, alpha_gen_det = self.sess.run(self.out_gen_det_impute, feed_dict)
        x_gen_det = self.sess.run(self.model_vars['x_hat'], {self.model_vars['a_seq']: a_gen_det,
                                                             self.ph_steps: self.test_data.timesteps})

        if plot:
            save_true_generated_frames(x_true, x_gen_det, self.config.log_dir + '/video_true_gen_%05d.mp4' % n)
            if self.config.dim_a == 2:
                plot_ball_trajectories_comparison(a_reconstr, a_gen_det, a_imputed,
                                                  self.config.log_dir + '/plot_imputation_%05d.png' % n,
                                                  nrows=4, ncols=4, mask=mask_impute)
            else:
                plot_auxiliary([a_reconstr, a_gen_det, a_imputed],
                               self.config.log_dir + '/plot_imputation_%05d.png' % n)

        # For a more fair comparison against pure generation only look at time steps with no observed variables
        mask_unobs = mask_impute < 0.5
        x_true_unobs = x_true[mask_unobs]

        # Get hamming distance on unobserved variables
        ham_unobs = dict()
        for key, value in zip(('gen', 'filt', 'smooth'), (x_gen_det, x_filtered, x_imputed)):
            ham_unobs[key] = hamming(x_true_unobs.flatten() > 0.5, value[mask_unobs].flatten() > 0.5)

        # Baseline is considered as the biggest hamming distance between two frames in the data
        hamming_baseline = 0.0
        for i in [0, 3, 6]:
            for j in [9, 12, 15]:
                tmp_dist = hamming((x_true[0, i] > 0.5).flatten(), (x_true[0, j] > 0.5).flatten())
                hamming_baseline = np.max([hamming_baseline, tmp_dist])

        # Return results
        a_reconstr_unobs = a_reconstr[mask_unobs]
        norm_rmse_a_imputed = norm_rmse(a_imputed[mask_unobs], a_reconstr_unobs)
        norm_rmse_a_gen_det = norm_rmse(a_gen_det[mask_unobs], a_reconstr_unobs)

        if plot:
            print("Hamming distance. x_imputed: %.5f, x_filtered: %.5f, x_gen_det: %.5f, baseline: %.5f. " % (
                ham_unobs['smooth'], ham_unobs['filt'], ham_unobs['gen'], hamming_baseline))
            print("Normalized RMSE. a_imputed: %.3f, a_gen_det: %.3f" % (norm_rmse_a_imputed, norm_rmse_a_gen_det))

        out_res = (ham_unobs['smooth'], ham_unobs['filt'], ham_unobs['gen'],
                   hamming_baseline, norm_rmse_a_imputed, norm_rmse_a_gen_det)
        return out_res

    def img_alpha_nn(self, range_x=(-30, 30), range_y=(-30, 30), N_points=50, n=99999):
        """ Visualise the output of the dynamics parameter network alpha over _a_ when dim_a == 2 and alpha_rnn=False
        
        :param range_x: range of first dimension of a
        :param range_y: range of second dimension of a
        :param N_points: points to sample
        :param n: epoch number
        :return: None
        """
        x = np.linspace(range_x[0], range_x[1], N_points)
        y = np.linspace(range_y[0], range_y[1], N_points)
        xv, yv = np.meshgrid(x, y)

        f, ax = plt.subplots(1, self.config.K, figsize=(18, 6))
        for k in range(self.config.K):
            out = np.zeros_like(xv)
            for i in range(N_points):
                for j in range(N_points):
                    a_prev = np.expand_dims(np.array([xv[i, j], yv[i, j]]), 0)
                    alpha_out = self.sess.run(self.out_alpha, {self.a_prev: a_prev})
                    out[i, j] = alpha_out[0][k]

            np.save(self.config.log_dir + '/image_alpha_%05d_%d' % (n, k), out)

            ax[k].pcolor(xv, yv, out, cmap='Greys')
            ax[k].set_aspect(1)
            ax[k].set_yticks([])
            ax[k].set_xticks([])

        plt.savefig(self.config.log_dir + '/image_alpha_%05d.png' % n, format='png', bbox_inches='tight', dpi=80)
        plt.close()

    def mask_impute_planning(self, t_init_mask=4, t_steps_mask=12):
        """ Create mask with missing values in the middle of the sequence
        :param t_init_mask: observed steps in the beginning of the sequence
        :param t_steps_mask: observed steps in the end
        :return: np.ndarray
        """
        mask_impute = np.ones((self.config.batch_size, self.test_data.timesteps), dtype=np.float32)
        t_end_mask = t_init_mask + t_steps_mask
        mask_impute[:, t_init_mask: t_end_mask] = 0.0
        return mask_impute

    def mask_impute_random(self, t_init_mask=4, drop_prob=0.5):
        """ Create mask with values missing at random
        :param t_init_mask: observed steps in the beginning of the sequence
        :param drop_prob: probability of not observing a step
        :return: np.ndarray
        """
        mask_impute = np.ones((self.config.batch_size, self.test_data.timesteps), dtype=np.float32)

        n_steps = self.test_data.timesteps - t_init_mask
        mask_impute[:, t_init_mask:] = np.random.choice([0, 1], size=(self.config.batch_size, n_steps),
                                                                   p=[drop_prob, 1.0 - drop_prob])
        return mask_impute

    def impute_all(self, mask_impute, t_init_mask, n=99999, plot=True):
        """ Iterate over batches in the test set
        :param mask_impute: mask to apply
        :param t_init_mask: observed steps in the beginning of the sequence
        :param n: epoch number
        :param plot: Save plots
        :return: average of imputation errors
        """
        results = []
        for i in range(self.test_data.sequences // self.config.batch_size):
            results.append(self.impute(mask_impute, t_init_mask=t_init_mask, idx_batch=i, n=n, plot=plot))
        return np.array(results).mean(axis=0)

    def imputation_plot(self, mask_type):
        """ Generate imputation plots for varying levels of observed data
        :param mask_type: str, missing or planning
        :return: None
        """
        time_imput_start = time.time()
        out_res_all = []

        if mask_type == 'missing_planning':
            vec = range(1, 17, 2)
            xlab = 'Number of unobserved steps'
        else:
            vec = np.linspace(0.1, 1.0, num=10)
            xlab = 'Drop probability'

        for i, v in enumerate(vec):
            if mask_type == 'missing_planning':
                print("--- Imputation planning, t_steps_mask %s" % v)
                mask_impute = self.mask_impute_planning(t_init_mask=self.config.t_init_mask,
                                                        t_steps_mask=v)
            elif mask_type == 'missing_random':
                print("--- Imputation random, drop_prob %s" % v)
                mask_impute = self.mask_impute_random(t_init_mask=self.config.t_init_mask,
                                                      drop_prob=v)
            else:
                raise NotImplementedError

            out_res = self.impute_all(mask_impute, t_init_mask=self.config.t_init_mask,
                                                       plot=False, n=100+i)
            out_res_all.append(out_res)

        out_res_all = np.array(out_res_all)
        hamm_x_imputed = out_res_all[:, 0]
        hamm_x_filtered = out_res_all[:, 1]
        baseline = out_res_all[:, 3]

        results = [(baseline, 'Baseline'),
                   (hamm_x_imputed, 'KVAE smoothing'),
                   (hamm_x_filtered, 'KVAE filtering')]

        print(out_res_all)
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.rcParams['xtick.labelsize'] = 14
        mpl.rcParams['ytick.labelsize'] = 14
        for dist, label in results:
            if label == 'Baseline':
                linestyle = '--'
            else:
                linestyle = '.-'
            plt.plot(vec, dist, linestyle, linewidth=3, ms=20, label=label)
        plt.xlabel(xlab, fontsize=20)
        plt.ylabel('Hamming distance', fontsize=20)
        plt.legend(fontsize=20, loc=1)
        plt.savefig(self.config.log_dir + '/imputation_%s.png' % mask_type)
        plt.close()

        np.savez(self.config.log_dir + '/imputation_results_%s'% mask_type, results=results)
        print('Imputation plot  took %.2fs' % (time.time()-time_imput_start))

    @staticmethod
    def def_summary(prefix, elbo_tot, elbo_kf, kf_log_probs, elbo_vae, log_px, log_qa):
        """ Add ELBO terms to a TF Summary object for Tensorboard
        """
        mean_kf_log_probs = np.mean(kf_log_probs, axis=0)

        summary = tf.Summary()
        summary.value.add(tag=prefix + '_elbo_tot', simple_value=np.mean(elbo_tot))
        summary.value.add(tag=prefix + '_elbo_kf', simple_value=np.mean(elbo_kf))
        summary.value.add(tag=prefix + '_elbo_vae', simple_value=np.mean(elbo_vae))
        summary.value.add(tag=prefix + '_vae_px', simple_value=np.mean(log_px))
        summary.value.add(tag=prefix + '_vae_qa', simple_value=np.mean(log_qa))
        summary.value.add(tag=prefix + '_kf_transitions', simple_value=mean_kf_log_probs[0])
        summary.value.add(tag=prefix + '_kf_emissions', simple_value=mean_kf_log_probs[1])
        summary.value.add(tag=prefix + '_kf_init', simple_value=mean_kf_log_probs[2])
        summary.value.add(tag=prefix + '_kf_entropy', simple_value=mean_kf_log_probs[3])

        return summary
