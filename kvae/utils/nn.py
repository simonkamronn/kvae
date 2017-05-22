import tensorflow as tf
from tensorflow.python.framework import constant_op
import numpy as np

const_log_pdf = (- 0.5 * np.log(2 * np.pi)).astype('float32')


def dclip(x, min, max):
    return x + tf.stop_gradient(tf.clip_by_value(x, min, max) - x)


def simple_sample(mu, var):
    epsilon = tf.random_normal(tf.shape(var), name="epsilon")
    return mu + tf.sqrt(var) * epsilon


def log_gaussian(x, mean, var):
    # return - 0.5 * tf.log(2*np.pi) - var / 2 - tf.square((x - mean)) / (2 * tf.exp(var) + 1e-8)
    return const_log_pdf - tf.log(var) / 2 - tf.square(x - mean) / (2 * var)


def log_bernoulli(x, p, eps=0.0):
    p = tf.clip_by_value(p, eps, 1.0 - eps)
    return x * tf.log(p) + (1 - x) * tf.log(1 - p)


def kl(mean, var):
    # return - 0.5 * (1 + var - tf.square(mean) - tf.exp(var))
    return -0.5 * (1 + tf.log(var) - tf.square(mean) - var)


def log_likelihood(mu, var, x, muq, varq, a, mask_flat, config):
    if config.out_distr == 'bernoulli':
        log_lik = log_bernoulli(x, mu, eps=1e-6)  # (bs*L, d1*d2)
    elif config.out_distr == 'gaussian':
        log_lik = log_gaussian(x, mu, var)

    log_lik = tf.reduce_sum(log_lik, 1)  # (bs*L, )
    log_lik = tf.multiply(mask_flat, log_lik)
    # TODO: dropout scales the output as input/keep_prob. Issue?
    if config.ll_keep_prob < 1.0:
        log_lik = tf.layers.dropout(log_lik, config.ll_keep_prob)

    # We compute the log-likelihood *per frame*
    num_el = tf.reduce_sum(mask_flat)
    log_px_given_a = tf.truediv(tf.reduce_sum(log_lik), num_el)  # ()

    if config.use_vae:
        log_qa_given_x = tf.reduce_sum(log_gaussian(a, muq, varq), 1)  # (bs*L, )
        log_qa_given_x = tf.multiply(mask_flat, log_qa_given_x)
        log_qa_given_x = tf.truediv(tf.reduce_sum(log_qa_given_x), num_el)  # ()
    else:
        log_qa_given_x = tf.constant(0.0, dtype=tf.float32, shape=())

    LL = log_px_given_a - log_qa_given_x
    return LL, log_px_given_a, log_qa_given_x


def norm_rmse(predictions, targets):
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    norm_rmse = rmse / np.std(targets)
    return norm_rmse


def get_activation_fn(name):
    # Get activation function for hidden layers
    if name.lower() == 'relu':
        activation_fn = tf.nn.relu
    elif name.lower() == 'tanh':
        activation_fn = tf.nn.tanh
    elif name.lower() == 'elu':
        activation_fn = tf.nn.elu
    else:
        activation_fn = None
    return activation_fn


def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        # k = tf.shape(logits)[-1]
        # y_hard = tf.cast(tf.one_hot(tf.argmax(y, 1), k), y.dtype)
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def kl_gumbel(logits, N, K):
    q_y = tf.nn.softmax(logits)
    log_q_y = tf.log(q_y + 1e-20)
    return tf.reduce_sum(tf.reshape(q_y * (log_q_y - tf.log(1.0/K)), [-1, N, K]), [1, 2])


class IdentityInitializer(object):
    def __init__(self, dtype=tf.float32):
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        if len(shape) == 1:
            return constant_op.constant(0., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]:
            return constant_op.constant(np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]:
            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0]/2, shape[1]/2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return constant_op.constant(array, dtype=dtype)
        else:
            constant_op.constant(0., dtype=dtype, shape=shape)


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    filters = c // r ** 2

    bsize = tf.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
    X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
    X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def ps(X, r, channels=1):
    if channels > 1:
        Xc = tf.split(X, channels, 3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], 3)
    else:
        X = _phase_shift(X, r)
    return X


def subpixel_reshape(x, factor):
    """
    Reshape function for subpixel upsampling
    x: tensorflow tensor, shape = (bs,h,w,c)
    factor: interger, upsample factor
    Return: tensorflow tensor, shape = (bs,h*factor,w*factor,c//factor**2)
    """

    # input and output shapes
    bs, ih, iw, ic = x.get_shape().as_list()
    oh, ow, oc = ih * factor, iw * factor, ic // factor ** 2

    assert ic % factor == 0, "Number of input channels must be divisible by factor"

    intermediateshp = (-1, iw, iw, oc, factor, factor)
    x = tf.reshape(x, intermediateshp)
    x = tf.transpose(x, (0, 1, 4, 2, 5, 3))
    x = tf.reshape(x, (-1, oh, ow, oc))
    return x
