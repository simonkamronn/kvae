import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalTriL
import numpy as np


class KalmanFilter(object):
    """
    This class defines a Kalman Filter (Linear Gaussian State Space model), possibly with a dynamics parameter
    network alpha.
    """

    def __init__(self, dim_z, dim_y, dim_u=0, dim_k=1, **kwargs):

        self.dim_z = dim_z
        self.dim_y = dim_y
        self.dim_u = dim_u
        self.dim_k = dim_k

        # Initializer for identity matrix
        self.eye_init = lambda shape, dtype=np.float32: np.eye(*shape, dtype=dtype)

        # Pop all variables
        init = kwargs.pop('mu', np.zeros((dim_z, ), dtype=np.float32))
        self.mu = tf.get_variable('mu', initializer=init, trainable=False)  # state

        init = kwargs.pop('Sigma', self.eye_init((dim_z, dim_z))).astype(np.float32)
        self.Sigma = tf.get_variable('Sigma', initializer=init, trainable=False)  # uncertainty covariance

        init = kwargs.pop('y_0', np.zeros((dim_y,))).astype(np.float32)
        self.y_0 = tf.get_variable('y_0', initializer=init)  # initial output

        init = kwargs.pop('A', self.eye_init((dim_z, dim_z)))
        self.A = tf.get_variable('A', initializer=init)

        init = kwargs.pop('B', self.eye_init((dim_z, dim_u))).astype(np.float32)
        self.B = tf.get_variable('B', initializer=init)  # control transition matrix

        init = kwargs.pop('Q', self.eye_init((dim_z, dim_z))).astype(np.float32)
        self.Q = tf.get_variable('Q', initializer=init, trainable=False)  # process uncertainty

        init = kwargs.pop('C', self.eye_init((dim_y, dim_z))).astype(np.float32)
        self.C = tf.get_variable('C', initializer=init)   # Measurement function

        init = kwargs.pop('R', self.eye_init((dim_y, dim_y))).astype(np.float32)
        self.R = tf.get_variable('R', initializer=init, trainable=False)   # state uncertainty

        self._alpha_sq = tf.constant(1., dtype=tf.float32) # fading memory control
        self.M = 0              # process-measurement cross correlation

        # identity matrix
        self._I = tf.constant(self.eye_init((dim_z, dim_z)), name='I')

        # Get variables that are possibly defined with tensors
        self.y = kwargs.pop('y', None)
        if self.y is None:
            self.y = tf.placeholder(tf.float32, shape=(None, None, dim_y), name='y')

        self.u = kwargs.pop('u', None)
        if self.u is None:
            self.u = tf.placeholder(tf.float32, shape=(None, None, dim_u), name='u')

        self.mask = kwargs.pop('mask', None)
        if self.mask is None:
            self.mask = tf.placeholder(tf.float32, shape=(None, None), name='mask')

        self.alpha = kwargs.pop('alpha', None)
        self.state = kwargs.pop('state', None)
        self.log_likelihood = None

    def forward_step_fn(self, params, inputs):
        """
        Forward step over a batch, to be used in tf.scan
        :param params:
        :param inputs: (batch_size, variable dimensions)
        :return:
        """
        mu_pred, Sigma_pred, _, _, alpha, u, state, buffer, _, _, _ = params
        y = tf.slice(inputs, [0, 0], [-1, self.dim_y])  # (bs, dim_y)
        _u = tf.slice(inputs, [0, self.dim_y], [-1, self.dim_u])  # (bs, dim_u)
        mask = tf.slice(inputs, [0, self.dim_y + self.dim_u], [-1, 1])  # (bs, dim_u)

        # Mixture of C
        C = tf.matmul(alpha, tf.reshape(self.C, [-1, self.dim_y*self.dim_z]))  # (bs, k) x (k, dim_y*dim_z)
        C = tf.reshape(C, [-1, self.dim_y, self.dim_z])  # (bs, dim_y, dim_z)
        C.set_shape([Sigma_pred.get_shape()[0], self.dim_y, self.dim_z])

        # Residual
        y_pred = tf.squeeze(tf.matmul(C, tf.expand_dims(mu_pred, 2)))  # (bs, dim_y)
        r = y - y_pred  # (bs, dim_y)

        # project system uncertainty into measurement space
        S = tf.matmul(tf.matmul(C, Sigma_pred), C, transpose_b=True) + self.R  # (bs, dim_y, dim_y)

        S_inv = tf.matrix_inverse(S)
        K = tf.matmul(tf.matmul(Sigma_pred, C, transpose_b=True), S_inv)  # (bs, dim_z, dim_y)

        # For missing values, set to 0 the Kalman gain matrix
        K = tf.multiply(tf.expand_dims(mask, 2), K)

        # Get current mu and Sigma
        mu_t = mu_pred + tf.squeeze(tf.matmul(K, tf.expand_dims(r, 2)))  # (bs, dim_z)
        I_KC = self._I - tf.matmul(K, C)  # (bs, dim_z, dim_z)
        Sigma_t = tf.matmul(tf.matmul(I_KC, Sigma_pred), I_KC, transpose_b=True) + self._sast(self.R, K)  # (bs, dim_z, dim_z)

        # Mixture of A
        alpha, state, u, buffer = self.alpha(tf.multiply(mask, y) + tf.multiply((1-mask), y_pred), state, _u, buffer, reuse=True)  # (bs, k)
        A = tf.matmul(alpha, tf.reshape(self.A, [-1, self.dim_z*self.dim_z]))  # (bs, k) x (k, dim_z*dim_z)
        A = tf.reshape(A, [-1, self.dim_z, self.dim_z])  # (bs, dim_z, dim_z)
        A.set_shape(Sigma_pred.get_shape())  # set shape to batch_size x dim_z x dim_z

        # Mixture of B
        B = tf.matmul(alpha, tf.reshape(self.B, [-1, self.dim_z*self.dim_u]))  # (bs, k) x (k, dim_y*dim_z)
        B = tf.reshape(B, [-1, self.dim_z, self.dim_u])  # (bs, dim_y, dim_z)
        B.set_shape([A.get_shape()[0], self.dim_z, self.dim_u])

        # Prediction
        mu_pred = tf.squeeze(tf.matmul(A, tf.expand_dims(mu_t, 2))) + tf.squeeze(tf.matmul(B, tf.expand_dims(u, 2)))
        Sigma_pred = tf.scalar_mul(self._alpha_sq, tf.matmul(tf.matmul(A, Sigma_t), A, transpose_b=True) + self.Q)

        return mu_pred, Sigma_pred, mu_t, Sigma_t, alpha, u, state, buffer, A, B, C

    def backward_step_fn(self, params, inputs):
        """
        Backwards step over a batch, to be used in tf.scan
        :param params:
        :param inputs: (batch_size, variable dimensions)
        :return:
        """
        mu_back, Sigma_back = params
        mu_pred_tp1, Sigma_pred_tp1, mu_filt_t, Sigma_filt_t, A = inputs

        # J_t = tf.matmul(tf.reshape(tf.transpose(tf.matrix_inverse(Sigma_pred_tp1), [0, 2, 1]), [-1, self.dim_z]),
        #                 self.A)
        # J_t = tf.transpose(tf.reshape(J_t, [-1, self.dim_z, self.dim_z]), [0, 2, 1])
        J_t = tf.matmul(tf.transpose(A, [0, 2, 1]), tf.matrix_inverse(Sigma_pred_tp1))
        J_t = tf.matmul(Sigma_filt_t, J_t)

        mu_back = mu_filt_t + tf.matmul(J_t, mu_back - mu_pred_tp1)
        Sigma_back = Sigma_filt_t + tf.matmul(J_t, tf.matmul(Sigma_back - Sigma_pred_tp1, J_t, adjoint_b=True))

        return mu_back, Sigma_back

    def compute_forwards(self, reuse=None):
        """Compute the forward step in the Kalman filter.
           The forward pass is intialized with p(z_1)=N(self.mu, self.Sigma).
           We then return the mean and covariances of the predictive distribution p(z_t|z_tm1,u_t), t=2,..T+1
           and the filtering distribution p(z_t|x_1:t,u_1:t), t=1,..T
           We follow the notation of Murphy's book, section 18.3.1
        """

        # To make sure we are not accidentally using the real outputs in the steps with missing values, set them to 0.
        y_masked = tf.multiply(tf.expand_dims(self.mask, 2), self.y)
        inputs = tf.concat([y_masked, self.u, tf.expand_dims(self.mask, 2)], axis=2)

        y_prev = tf.expand_dims(self.y_0, 0)  # (1, dim_y)
        y_prev = tf.tile(y_prev, (tf.shape(self.mu)[0], 1))
        alpha, state, u, buffer = self.alpha(y_prev, self.state, self.u[:, 0], init_buffer=True, reuse= reuse)

        # dummy matrix to initialize B and C in scan
        dummy_init_A = tf.ones([self.Sigma.get_shape()[0], self.dim_z, self.dim_z])
        dummy_init_B = tf.ones([self.Sigma.get_shape()[0], self.dim_z, self.dim_u])
        dummy_init_C = tf.ones([self.Sigma.get_shape()[0], self.dim_y, self.dim_z])
        forward_states = tf.scan(self.forward_step_fn, tf.transpose(inputs, [1, 0, 2]),
                                 initializer=(self.mu, self.Sigma, self.mu, self.Sigma, alpha, u, state, buffer,
                                              dummy_init_A, dummy_init_B, dummy_init_C),
                                 parallel_iterations=1, name='forward')
        return forward_states

    def compute_backwards(self, forward_states):
        mu_pred, Sigma_pred, mu_filt, Sigma_filt, alpha, u, state, buffer, A, B, C = forward_states
        mu_pred = tf.expand_dims(mu_pred, 3)
        mu_filt = tf.expand_dims(mu_filt, 3)
        # The tf.scan below that does the smoothing is initialized with the filtering distribution at time T.
        # following the derivarion in Murphy's book, we then need to discard the last time step of the predictive
        # (that will then have t=2,..T) and filtering distribution (t=1:T-1)
        states_scan = [mu_pred[:-1, :, :, :],
                       Sigma_pred[:-1, :, :, :],
                       mu_filt[:-1, :, :, :],
                       Sigma_filt[:-1, :, :, :],
                       A[:-1]]

        # Reverse time dimension
        dims = [0]
        for i, state in enumerate(states_scan):
            states_scan[i] = tf.reverse(state, dims)

        # Compute backwards states
        backward_states = tf.scan(self.backward_step_fn, states_scan,
                                  initializer=(mu_filt[-1, :, :, :], Sigma_filt[-1, :, :, :]), parallel_iterations=1,
                                  name='backward')

        # Reverse time dimension
        backward_states = list(backward_states)
        dims = [0]
        for i, state in enumerate(backward_states):
            backward_states[i] = tf.reverse(state, dims)

        # Add the final state from the filtering distribution
        backward_states[0] = tf.concat([backward_states[0], mu_filt[-1:, :, :, :]], axis=0)
        backward_states[1] = tf.concat([backward_states[1], Sigma_filt[-1:, :, :, :]], axis=0)

        # Remove extra dimension in the mean
        backward_states[0] = backward_states[0][:, :, :, 0]

        return backward_states, A, B, C, alpha

    def sample_generative_tf(self, backward_states, n_steps, deterministic=True, init_fixed_steps=1):
        """
        Get a sample from the generative model
        """
        # Get states from the Kalman filter to get the initial state
        mu_z, sigma_z = backward_states
        # z = tf.contrib.distributions.MultivariateNormalTriL(mu_z[seq_idx, 0], sigma_z[seq_idx, 0]).sample()

        if init_fixed_steps > 0:
            z = mu_z[:, 0]
            z = tf.expand_dims(z, 2)
        else:
            raise("Prior sampling from z not implemented")

        if not deterministic:
            # Pre-compute samples of noise
            noise_trans = MultivariateNormalTriL(tf.zeros((self.dim_z,)), tf.cholesky(self.Q))
            epsilon = noise_trans.sample((z.get_shape()[0].value, n_steps))
            noise_emiss = MultivariateNormalTriL(tf.zeros((self.dim_y,)), tf.cholesky(self.R))
            delta = noise_emiss.sample((z.get_shape()[0].value, n_steps))
        else:
            epsilon = tf.zeros((z.get_shape()[0], n_steps, self.dim_z))
            delta = tf.zeros((z.get_shape()[0], n_steps, self.dim_y))

        y_prev = tf.expand_dims(self.y_0, 0)  # (1, dim_y)
        y_prev = tf.tile(y_prev, (tf.shape(self.mu)[0], 1))  # (bs, dim_y)
        alpha, state, u, buffer = self.alpha(y_prev, self.state, self.u[:, 0], reuse=True, init_buffer=True)

        y_samples = list()
        z_samples = list()
        alpha_samples = list()
        for n in range(n_steps):

            # Mixture of C
            C = tf.matmul(alpha, tf.reshape(self.C, [-1, self.dim_y*self.dim_z]))  # (bs, k) x (k, dim_y*dim_z)
            C = tf.reshape(C, [-1, self.dim_y, self.dim_z])  # (bs, dim_y, dim_z)

            # Output for the current time step
            y = tf.matmul(C, z) + tf.expand_dims(delta[:, n], 2)
            y = tf.squeeze(y, 2)

            # Store current state and output at time t
            z_samples.append(tf.squeeze(z, 2))
            y_samples.append(y)

            # Compute the mixture of A
            alpha, state, u, buffer = self.alpha(y, state, self.u[:, n], buffer, reuse=True)
            alpha_samples.append(alpha)
            A = tf.matmul(alpha, tf.reshape(self.A, [-1, self.dim_z * self.dim_z]))
            A = tf.reshape(A, [-1, self.dim_z, self.dim_z])

            # Mixture of B
            B = tf.matmul(alpha, tf.reshape(self.B, [-1, self.dim_z*self.dim_u]))  # (bs, k) x (k, dim_y*dim_z)
            B = tf.reshape(B, [-1, self.dim_z, self.dim_u])  # (bs, dim_y, dim_z)

            # Get new state z_{t+1}
            # z = tf.matmul(A, z) + tf.matmul(B,  tf.expand_dims(self.u[:, n],2)) + tf.expand_dims(epsilon[:, n], 2)
            if (n + 1) >= init_fixed_steps:
                z = tf.matmul(A, z) + tf.matmul(B,  tf.expand_dims(u, 2)) + tf.expand_dims(epsilon[:, n], 2)
            else:
                z = mu_z[:, n+1]
                z = tf.expand_dims(z, 2)

        return tf.stack(y_samples, 1), tf.stack(z_samples, 1), tf.stack(alpha_samples, 1)

    def get_elbo(self, backward_states, A, B, C):

        mu_smooth = backward_states[0]
        Sigma_smooth = backward_states[1]

        # Sample from smoothing distribution
        # jitter = 1e-2 * tf.eye(tf.shape(Sigma_smooth)[-1], batch_shape=tf.shape(Sigma_smooth)[0:-2])
        # mvn_smooth = tf.contrib.distributions.MultivariateNormalTriL(mu_smooth, Sigma_smooth + jitter)
        mvn_smooth = MultivariateNormalTriL(mu_smooth, tf.cholesky(Sigma_smooth))
        z_smooth = mvn_smooth.sample()

        ## Transition distribution \prod_{t=2}^T p(z_t|z_{t-1}, u_{t})
        # We need to evaluate N(z_t; Az_tm1 + Bu_t, Q), where Q is the same for all the elements
        # z_tm1 = tf.reshape(z_smooth[:, :-1, :], [-1, self.dim_z])
        # Az_tm1 = tf.transpose(tf.matmul(self.A, tf.transpose(z_tm1)))
        Az_tm1 = tf.reshape(tf.matmul(A[:, :-1], tf.expand_dims(z_smooth[:, :-1], 3)), [-1, self.dim_z])

        # Remove the first input as our prior over z_1 does not depend on it
        # u_t_resh = tf.reshape(u, [-1, self.dim_u])
        # Bu_t = tf.transpose(tf.matmul(self.B, tf.transpose(u_t_resh)))
        Bu_t = tf.reshape(tf.matmul(B[:, :-1], tf.expand_dims(self.u[:, 1:], 3)), [-1, self.dim_z])
        mu_transition = Az_tm1 + Bu_t
        z_t_transition = tf.reshape(z_smooth[:, 1:, :], [-1, self.dim_z])

        # MultivariateNormalTriL supports broadcasting only for the inputs, not for the covariance
        # To exploit this we then write N(z_t; Az_tm1 + Bu_t, Q) as N(z_t - Az_tm1 - Bu_t; 0, Q)
        trans_centered = z_t_transition - mu_transition
        mvn_transition = MultivariateNormalTriL(tf.zeros(self.dim_z), tf.cholesky(self.Q))
        log_prob_transition = mvn_transition.log_prob(trans_centered)

        ## Emission distribution \prod_{t=1}^T p(y_t|z_t)
        # We need to evaluate N(y_t; Cz_t, R). We write it as N(y_t - Cz_t; 0, R)
        # z_t_emission = tf.reshape(z_smooth, [-1, self.dim_z])
        # Cz_t = tf.transpose(tf.matmul(self.C, tf.transpose(z_t_emission)))
        Cz_t = tf.reshape(tf.matmul(C, tf.expand_dims(z_smooth, 3)), [-1, self.dim_y])

        y_t_resh = tf.reshape(self.y, [-1, self.dim_y])
        emiss_centered = y_t_resh - Cz_t
        mvn_emission = MultivariateNormalTriL(tf.zeros(self.dim_y), tf.cholesky(self.R))
        mask_flat = tf.reshape(self.mask, (-1, ))
        log_prob_emission = mvn_emission.log_prob(emiss_centered)
        log_prob_emission = tf.multiply(mask_flat, log_prob_emission)

        ## Distribution of the initial state p(z_1|z_0)
        z_0 = z_smooth[:, 0, :]
        mvn_0 = MultivariateNormalTriL(self.mu, tf.cholesky(self.Sigma))
        log_prob_0 = mvn_0.log_prob(z_0)

        # Entropy log(\prod_{t=1}^T p(z_t|y_{1:T}, u_{1:T}))
        entropy = - mvn_smooth.log_prob(z_smooth)
        entropy = tf.reshape(entropy, [-1])
        # entropy = tf.zeros(())

        # Compute terms of the lower bound
        # We compute the log-likelihood *per frame*
        num_el = tf.reduce_sum(mask_flat)
        log_probs = [tf.truediv(tf.reduce_sum(log_prob_transition), num_el),
                     tf.truediv(tf.reduce_sum(log_prob_emission), num_el),
                     tf.truediv(tf.reduce_sum(log_prob_0), num_el),
                     tf.truediv(tf.reduce_sum(entropy), num_el)]

        kf_elbo = tf.reduce_sum(log_probs)

        return kf_elbo, log_probs, z_smooth

    def filter(self):
        mu_pred, Sigma_pred, mu_filt, Sigma_filt, alpha, u, state, buffer, A, B, C = forward_states = \
            self.compute_forwards(reuse=True)
        forward_states = [mu_filt, Sigma_filt]
        # Swap batch dimension and time dimension
        forward_states[0] = tf.transpose(forward_states[0], [1, 0, 2])
        forward_states[1] = tf.transpose(forward_states[1], [1, 0, 2, 3])
        return tuple(forward_states), tf.transpose(A, [1, 0, 2, 3]), tf.transpose(B, [1, 0, 2, 3]), \
               tf.transpose(C, [1, 0, 2, 3]), tf.transpose(alpha, [1, 0, 2])

    def smooth(self):
        backward_states, A, B, C, alpha = self.compute_backwards(self.compute_forwards())
        # Swap batch dimension and time dimension
        backward_states[0] = tf.transpose(backward_states[0], [1, 0, 2])
        backward_states[1] = tf.transpose(backward_states[1], [1, 0, 2, 3])
        return tuple(backward_states), tf.transpose(A, [1, 0, 2, 3]), tf.transpose(B, [1, 0, 2, 3]), \
               tf.transpose(C, [1, 0, 2, 3]), tf.transpose(alpha, [1, 0, 2])

    def _sast(self, a, s):
        _, dim_1, dim_2 = s.get_shape().as_list()
        sast = tf.matmul(tf.reshape(s, [-1, dim_2]), a, transpose_b=True)
        sast = tf.transpose(tf.reshape(sast, [-1, dim_1, dim_2]), [0, 2, 1])
        sast = tf.matmul(s, sast)
        return sast

