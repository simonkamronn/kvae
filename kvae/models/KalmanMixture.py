from tensorflow.contrib import slim
from tensorflow.contrib.layers import optimize_loss
from kvae.filter import KalmanFilter
from kvae.utils.plotting import plot_auxiliary, plot_alpha_grid, plot_ball_trajectories
from kvae.utils.data import PymunkData
from kvae.utils.nn import *
from tensorflow.contrib.rnn import BasicLSTMCell
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})
np.random.seed(13837)


class KalmanMixture(object):
    """Soft switching linear gaussian state space model

    config: config from kvae.utils.get_config()
    """

    def __init__(self, config, sess):
        self.config = config

        self.train_data = PymunkData("../data/{}.npz".format(config.dataset), config)
        self.test_data = PymunkData("../data/{}_test.npz".format(config.dataset), config)
        self.state_dim = self.train_data.state_dim

        # Init variables
        A = np.array([np.eye(config.dim_z, dtype=np.float32).astype(np.float32)
                      for _ in range(config.K)])
        # A = np.array([config.init_kf_matrices * np.random.randn(config.dim_z, config.dim_z).astype(np.float32)
        #               for _ in range(config.K)])
        B = np.array([config.init_kf_matrices * np.random.randn(config.dim_z, config.dim_u).astype(np.float32)
                      for _ in range(config.K)])
        C = np.array([config.init_kf_matrices * np.random.randn(self.state_dim, config.dim_z).astype(np.float32)
                      for _ in range(config.K)])

        Q = config.noise_transition * np.eye(config.dim_z, dtype=np.float32)
        R = config.noise_emission * np.eye(self.state_dim, dtype=np.float32)

        mu = np.zeros((self.config.batch_size, config.dim_z), dtype=np.float32)
        Sigma = np.tile(config.init_cov * np.eye(config.dim_z, dtype=np.float32), (self.config.batch_size, 1, 1))

        a_0 = np.zeros((self.state_dim,), dtype=np.float32)  # Initial variable a_0

        # Collect initial variables
        self.init_vars = dict(A=A, B=B, C=C, Q=Q, R=R, mu=mu, Sigma=Sigma, a_0=a_0)

        # Set activation function
        self.activation_fn = get_activation_fn(config.activation)

        # Set Tensorflow session
        self.sess = sess

        # Init placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, None, self.state_dim], name='x')
        self.ph_steps = tf.placeholder(tf.int32, shape=(), name='n_step')
        self.mask = tf.placeholder(tf.float32, shape=(None, None), name='mask')
        self.a_prev = tf.placeholder(tf.float32, shape=[None, self.state_dim], name='a_prev')  # For alpha NN plotting

        # Init various
        self.saver = None
        self.kf = None
        self.kf_updates = None
        self.all_updates = None
        self.lb_vars = None
        self.model_vars = None
        self.n_steps_gen = None
        self.out_gen = None
        self.out_gen_det = None
        self.out_gen_det_impute = None
        self.out_alpha = None
        self.train_summary = None
        self.test_summary = None
        self.tau = tf.Variable(10.0, name='gumbel_temp')
        self.alpha_logits = None

    def alpha(self, inputs, state=None, u=None, buffer=None, reuse=None, init_buffer=False, name='alpha'):
        """The mixing vector alpha for mixing transitions in a state space model

        Args:
            inputs: tensor to condition mixing vector on
            state: previous state if using RNN network to model alpha
            u: pass-through variable if u is given
            reuse: `True` or `None`; if `True`, we go into reuse mode for this scope as
                    well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
            init_buffer: initialize buffer for a_t
            name: name of the scope

        Returns:
            alpha: mixing vector of dimension (batch size, K)
            state: new state
            u: either inferred u from model or pass-through
        """
        # Increase the number of hidden units if we also learn u
        num_units = self.config.alpha_units * 2 if self.config.learn_u else self.config.alpha_units

        # Overwrite input buffer
        if init_buffer:
            buffer = tf.zeros((tf.shape(inputs)[0], self.state_dim, self.config.fifo_size), dtype=tf.float32)

        with tf.variable_scope(name, reuse=reuse):
            if self.config.alpha_rnn:
                # rnn_cell = GRUCell(num_units)
                rnn_cell = BasicLSTMCell(num_units, reuse=reuse)
                output, state = rnn_cell(inputs, state)
            else:
                # Shift buffer
                buffer = tf.concat([buffer[:, :, 1:], tf.expand_dims(inputs, 2)], 2)
                output = slim.repeat(
                    tf.reshape(buffer, (tf.shape(inputs)[0], self.state_dim * self.config.fifo_size)),
                    self.config.alpha_layers, slim.fully_connected, num_units,
                    get_activation_fn(self.config.alpha_activation), scope='hidden')

            if self.config.alpha_gumbel:
                logits = slim.fully_connected(output[:, :self.config.alpha_units],
                                              self.config.K,
                                              activation_fn=None,
                                              scope='gumbel_logits')
                if self.alpha_logits is None:
                    # TODO: This is a bit of a hack
                    self.alpha_logits = logits
                alpha = gumbel_softmax(logits, self.tau)
            else:
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

        # If K == 1 simplify the graph
        alpha = self.alpha if self.config.K > 1 else lambda inputs, state, u, reuse, init_buffer: tf.ones(
            [tf.shape(self.x)[0], self.config.K])

        # # Initial RNN state
        # init_rnn_np = np.zeros((1, self.config.alpha_units * 2 if self.config.learn_u else self.config.alpha_units))
        # self.init_rnn = tf.get_variable('init_rnn', initializer=init_rnn_np.astype(np.float32), trainable=False)
        # state_init_rnn = tf.tile(self.init_rnn, (self.config.batch_size, 1))

        dummy_lstm = BasicLSTMCell(self.config.alpha_units * 2 if self.config.learn_u else self.config.alpha_units)
        state_init_rnn = dummy_lstm.zero_state(self.config.batch_size, tf.float32)

        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_z=self.config.dim_z,
                               dim_y=self.state_dim,
                               dim_u=self.config.dim_u,
                               dim_k=self.config.K,
                               A=self.init_vars['A'],  # state transition function
                               B=self.init_vars['B'],  # control matrix
                               C=self.init_vars['C'],  # Measurement function
                               R=self.init_vars['R'],  # measurement noise
                               Q=self.init_vars['Q'],  # process noise
                               y=self.x,  # output
                               u=None,
                               mask=self.mask,
                               mu=self.init_vars['mu'],
                               Sigma=self.init_vars['Sigma'],
                               y_0=self.init_vars['a_0'],
                               alpha=alpha,
                               state=state_init_rnn
                               )

        # Get state calculation
        smooth, A, B, C, alpha_plot = self.kf.smooth()

        # Get a from the prior z
        a_mu_pred = tf.matmul(C, tf.expand_dims(smooth[0], 2), transpose_b=True)
        a_mu_pred_seq = tf.reshape(a_mu_pred, tf.stack((-1, self.ph_steps, self.state_dim)))

        # Compute variables for generation
        self.n_steps_gen = self.config.n_steps_gen  # We sample for this many iterations,
        self.out_gen_det = self.kf.sample_generative_tf(smooth, self.n_steps_gen, deterministic=True)
        self.out_gen = self.kf.sample_generative_tf(smooth, self.n_steps_gen, deterministic=False)
        self.out_gen_det_impute = self.kf.sample_generative_tf(smooth,
                                                               self.test_data.timesteps,
                                                               deterministic=True,
                                                               init_fixed_steps=self.config.t_init_mask)

        self.out_alpha, _, _, _ = self.alpha(self.a_prev, state=state_init_rnn, u=None, init_buffer=True, reuse=True)

        # Collect generated model variables
        self.model_vars = dict(smooth=smooth, A=A, B=B, C=C, alpha_plot=alpha_plot, a_mu_pred_seq=a_mu_pred_seq)

        return self

    def build_loss(self):
        # KF loss
        elbo_kf, kf_log_probs, z_smooth = self.kf.get_elbo(self.model_vars['smooth'],
                                                           self.model_vars['A'],
                                                           self.model_vars['B'],
                                                           self.model_vars['C'])
        # Calc num_batches
        num_batches = self.train_data.sequences // self.config.batch_size

        # Decreasing learning rate
        global_step = slim.create_global_step()
        learning_rate = tf.train.exponential_decay(self.config.init_lr, global_step,
                                                   self.config.decay_steps * num_batches,
                                                   self.config.decay_rate, staircase=True)

        # Add alpha entropy cost
        # entropy_factor = tf.train.polynomial_decay(tf.Variable(0.0),
        #                                            global_step,
        #                                            decay_steps=self.config.num_epochs * num_batches,
        #                                            end_learning_rate=0.01)
        # elbo_kf = elbo_kf + entropy_factor * tf.reduce_sum(self.model_vars['alpha_plot']
        #                                                    * tf.log(self.model_vars['alpha_plot']))

        if self.config.alpha_gumbel:
            # Decrease gumbel temperature
            self.tau = tf.maximum(tf.train.inverse_time_decay(self.tau, global_step,
                                                              self.config.gumbel_decay_steps, 0.99), 0.2)
            # self.tau_ph = self.tau_ph * tf.train.piecewise_constant(global_step,
            #                                                   [100, 1000],
            #                                                   [0.0, 1.0, 1.0])
            tf.summary.scalar('gumbel_temp', self.tau)

            # Get Gumbel kl term
            alpha_gumbel_kl = tf.reduce_sum(kl_gumbel(self.alpha_logits, 1, self.config.K))
        else:
            alpha_gumbel_kl = tf.constant(0., dtype=tf.float32)

        # Collect variables to monitor lb
        self.lb_vars = [elbo_kf, kf_log_probs, alpha_gumbel_kl]

        # Get list of vars
        kf_vars = [self.kf.A, self.kf.B, self.kf.C, self.kf.y_0]
        # alpha_vars = slim.get_variables('alpha') + [self.init_rnn]
        all_vars = tf.trainable_variables()

        self.kf_updates = optimize_loss(loss=-elbo_kf,
                                        global_step=global_step,
                                        learning_rate=learning_rate,
                                        optimizer='Adam',
                                        clip_gradients=self.config.max_grad_norm,
                                        variables=kf_vars,
                                        summaries=["gradients", "gradient_norm", "learning_rate"],
                                        name='kf_updates')

        self.all_updates = optimize_loss(loss=-elbo_kf + alpha_gumbel_kl,
                                           global_step=global_step,
                                           learning_rate=learning_rate,
                                           optimizer='Adam',
                                           clip_gradients=self.config.max_grad_norm,
                                           variables=all_vars,
                                           summaries=["gradients", "gradient_norm", "learning_rate"],
                                           name='all_updates')
        return self

    def initialize_variables(self):
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
        sess = self.sess

        writer = tf.summary.FileWriter(self.config.log_dir, sess.graph)
        mask_train = np.ones((self.config.batch_size, self.train_data.timesteps), dtype=np.float32)
        merged_summaries = tf.summary.merge_all()

        for n in range(self.config.num_epochs):
            # Shuffle training data
            self.train_data.shuffle(shuffle_images=False)
            # Noise to add
            noise = 0.0 *np.sqrt(self.config.noise_emission) * np.random.randn(self.train_data.sequences,
                                                                          self.train_data.timesteps,
                                                                          self.train_data.state_dim)

            elbo_kf = []
            kf_log_probs = []
            time_epoch_start = time.time()
            for i in range(self.train_data.sequences // self.config.batch_size):
                slc = slice(i * self.config.batch_size, (i + 1) * self.config.batch_size)
                feed_dict = {self.x: self.train_data.state[slc] + noise[slc],
                             self.kf.u: self.train_data.controls[slc],
                             self.mask: mask_train,
                             self.ph_steps: self.train_data.timesteps}

                if n < self.config.kf_update_steps:
                    sess.run(self.kf_updates, feed_dict)
                else:
                    sess.run(self.all_updates, feed_dict)

                # Bookkeeping.
                _elbo_kf, _kf_log_probs, _alpha_kl = sess.run(self.lb_vars, feed_dict)
                elbo_kf.append(_elbo_kf)
                kf_log_probs.append(_kf_log_probs)

            # Write to summary
            summary_train = self.def_summary('train', elbo_kf, kf_log_probs)
            writer.add_summary(summary_train, n)
            writer.add_summary(sess.run(merged_summaries, feed_dict), n)

            # Write out losses
            if (n + 1) % self.config.display_step == 0:
                mean_kf_log_probs = np.mean(kf_log_probs, axis=0)
                print("Epoch %d, ELBO: %.4f, log_probs [%.2f, %.2f, %.2f, %.2f], took %.2fs"
                      % (n, np.mean(elbo_kf), mean_kf_log_probs[0], mean_kf_log_probs[1], mean_kf_log_probs[2],
                         mean_kf_log_probs[3], time.time() - time_epoch_start))

            # Make illustrations
            if ((n + 1) % self.config.generate_step == 0) or (n == self.config.num_epochs - 1) or (n == 0):
                norm_rmse_a_imputed = self.impute(t_init_mask=self.config.t_init_mask,
                                                  t_steps_mask=self.config.t_steps_mask, n=n)
                self.generate(n=n)

                # We can only show the image for alpha when using a simple neural network
                if self.config.fifo_size == 1 and self.config.alpha_rnn is False and self.config.learn_u is False:
                    self.img_alpha_nn(n=n, range_x=(-16, 16), range_y=(-16, 16))

                # Test on previously unseen data
                test_elbo, summary_test = self.test()
                writer.add_summary(summary_test, n)

        # Save the last model. Should we use early stopping?
        self.saver.save(sess, self.config.log_dir + '/model.ckpt')
        neg_lower_bound = -np.mean(test_elbo)
        print("Negative lower_bound on the test set: %s" % neg_lower_bound)
        return norm_rmse_a_imputed

    def test(self):
        mask_test = np.ones((self.config.batch_size, self.test_data.timesteps), dtype=np.float32)

        elbo_kf = []
        kf_log_probs = []
        time_test_start = time.time()
        for i in range(self.test_data.sequences // self.config.batch_size):
            slc = slice(i * self.config.batch_size, (i + 1) * self.config.batch_size)
            feed_dict = {self.x: self.test_data.state[slc],
                         self.kf.u: self.test_data.controls[slc],
                         self.mask: mask_test,
                         self.ph_steps: self.test_data.timesteps}

            # Bookkeeping.
            _elbo_kf, _kf_log_probs, _alpha_kl = self.sess.run(self.lb_vars, feed_dict)
            elbo_kf.append(_elbo_kf)
            kf_log_probs.append(_kf_log_probs)

        # Write to summary
        summary = self.def_summary('test', elbo_kf, kf_log_probs)
        mean_kf_log_probs = np.mean(kf_log_probs, axis=0)
        print("-- TEST, ELBO %.2f, log_probs [%.2f, %.2f, %.2f, %.2f], took %.2fs"
              % (np.mean(elbo_kf), mean_kf_log_probs[0], mean_kf_log_probs[1],
                 mean_kf_log_probs[2], mean_kf_log_probs[3], time.time() - time_test_start))
        return np.mean(elbo_kf), summary

    def generate(self, idx_batch=0, n=99999):
        # Get initial state z_1
        mask_test = np.ones((self.config.batch_size, self.test_data.timesteps), dtype=np.float32)
        slc = slice(idx_batch * self.config.batch_size, (idx_batch + 1) * self.config.batch_size)
        feed_dict = {self.x: self.test_data.state[slc],
                     self.kf.u: self.test_data.controls[slc],
                     self.ph_steps: self.test_data.timesteps,
                     self.mask: mask_test}
        smooth_z = self.sess.run(self.model_vars['smooth'], feed_dict)

        # Sample deterministic generation
        feed_dict = {self.model_vars['smooth']: smooth_z,
                     self.kf.u: np.zeros((self.config.batch_size, self.n_steps_gen, self.config.dim_u)),
                     self.ph_steps: self.n_steps_gen}
        a_gen_det, _, alpha_gen_det = self.sess.run(self.out_gen_det, feed_dict)

        # Save movie
        plot_auxiliary([self.test_data.state[slc], a_gen_det],
                       self.config.log_dir + '/plot_generation_det_%05d.png' % n)
        plot_alpha_grid(alpha_gen_det, self.config.log_dir + '/alpha_generation_det_%05d.png' % n)

        # Sample stochastic
        a_gen, _, alpha_gen = self.sess.run(self.out_gen, feed_dict)

        # Save movie
        plot_auxiliary([self.test_data.state[slc], a_gen], self.config.log_dir + '/plot_generation_%05d.png' % n)
        plot_alpha_grid(alpha_gen, self.config.log_dir + '/alpha_generation_%05d.png' % n)

        # Save trajectory from a
        plot_ball_trajectories(a_gen_det, self.config.log_dir + '/plot_generation_balls_%05d.png' % n)

    def impute(self, t_init_mask=4, t_steps_mask=12, idx_batch=0, n=99999):
        """Impute missing data in a sequence
        """
        mask_impute = np.ones((self.config.batch_size, self.test_data.timesteps), dtype=np.float32)
        t_end_mask = t_init_mask + t_steps_mask
        mask_impute[:, t_init_mask:t_end_mask] = 0.0
        print(mask_impute[0, :])

        slc = slice(idx_batch * self.config.batch_size, (idx_batch + 1) * self.config.batch_size)
        feed_dict = {self.x: self.test_data.state[slc],
                     self.kf.u: self.test_data.controls[slc],
                     self.ph_steps: self.test_data.timesteps,
                     self.mask: mask_impute}

        # Compute reconstructions and imputations
        a_imputed, alpha_reconstr, smooth_z = self.sess.run([self.model_vars['a_mu_pred_seq'],
                                                             self.model_vars['alpha_plot'],
                                                             self.model_vars['smooth']],
                                                            feed_dict)

        str_imputation = "%s-%s-%s" % (t_init_mask, t_steps_mask, self.test_data.timesteps - t_steps_mask - t_init_mask)
        plot_auxiliary([self.test_data.state[slc], a_imputed],
                       self.config.log_dir + '/plot_imputation_%05d_%s.png' % (n, str_imputation))
        plot_alpha_grid(alpha_reconstr, self.config.log_dir + '/alpha_reconstr_%05d.png' % n)

        # Plot z_mu
        plot_auxiliary([smooth_z[0]], self.config.log_dir + '/plot_z_mu_smooth_%05d.png' % n)

        # Sample deterministic generation having access to the first t_init_mask frames for comparison
        # Get initial state z_1
        feed_dict = {self.x: self.test_data.state[slc][:, 0:t_init_mask],
                     self.kf.u: self.test_data.controls[slc][:, 0:t_init_mask],
                     self.ph_steps: t_init_mask,
                     self.mask: mask_impute[:, 0:t_init_mask]}
        smooth_z_gen = self.sess.run(self.model_vars['smooth'], feed_dict)
        feed_dict = {self.model_vars['smooth']: smooth_z_gen,
                     self.kf.u: self.test_data.controls[slc],
                     self.ph_steps: self.test_data.timesteps}
        a_gen_det, _, alpha_gen_det = self.sess.run(self.out_gen_det_impute, feed_dict)

        plot_auxiliary([self.test_data.state[slc], a_imputed, a_gen_det],
                       self.config.log_dir + '/plot_all_%05d_%s.png' % (n, str_imputation))
        # For a more fair comparison against pure generation only look at time steps with no observed variables
        norm_rmse_a_imputed = norm_rmse(a_imputed[:, t_init_mask:t_end_mask, :],
                                        self.test_data.state[slc][:, t_init_mask:t_end_mask])
        norm_rmse_a_gen_det = norm_rmse(a_gen_det[:, t_init_mask:t_end_mask, :],
                                        self.test_data.state[slc][:, t_init_mask:t_end_mask])

        print("Normalized RMSE. a_imputed: %.3f, a_gen_det: %.3f" % (norm_rmse_a_imputed, norm_rmse_a_gen_det))
        return norm_rmse_a_imputed

    def img_alpha_nn(self, range_x=(-16, 16), range_y=(-16, 16), N_points=50, n=99999):
        x = np.linspace(range_x[0], range_x[1], N_points)
        y = np.linspace(range_y[0], range_y[1], N_points)
        xv, yv = np.meshgrid(x, y)

        f, ax = plt.subplots(1, self.config.K, figsize=(18, 18))

        for k in range(self.config.K):
            out = np.zeros_like(xv)
            for i in range(N_points):
                for j in range(N_points):
                    a_prev = np.expand_dims(np.array([xv[i, j], yv[i, j]]), 0)
                    alpha_out = self.sess.run(self.out_alpha, {self.a_prev: a_prev})
                    out[i, j] = alpha_out[0][k]

            ax[k].pcolor(xv, yv, out, cmap='jet')
            ax[k].set_aspect(1)
        plt.savefig(self.config.log_dir + '/image_alpha_%05d.png' % n, format='png', bbox_inches='tight', dpi=80)

    @staticmethod
    def def_summary(prefix, elbo_kf, kf_log_probs):

        mean_kf_log_probs = np.mean(kf_log_probs, axis=0)
        summary = tf.Summary()
        summary.value.add(tag=prefix + '_elbo_kf', simple_value=np.mean(elbo_kf))
        summary.value.add(tag=prefix + '_kf_transitions', simple_value=mean_kf_log_probs[0])
        summary.value.add(tag=prefix + '_kf_emissions', simple_value=mean_kf_log_probs[1])
        summary.value.add(tag=prefix + '_kf_init', simple_value=mean_kf_log_probs[2])
        summary.value.add(tag=prefix + '_kf_entropy', simple_value=mean_kf_log_probs[3])

        return summary
