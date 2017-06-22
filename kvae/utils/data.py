import numpy as np


class PymunkData(object):
    """ Load sequences of images

    """
    def __init__(self, file_path, config):
        # Load data
        npzfile = np.load(file_path)
        self.images = npzfile['images'].astype(np.float32)
        if config.out_distr == 'bernoulli':
            self.images = (self.images > 0).astype('float32')

        # Load ground truth position and velocity (if present). This is not used in the KVAE experiments in the paper.
        if 'state' in npzfile:
            # Only load the position, not velocity
            self.state = npzfile['state'].astype(np.float32)[:, :, :2]
            self.velocity = npzfile['state'].astype(np.float32)[:, :, 2:]

            # Normalize the mean
            self.state = self.state - self.state.mean(axis=(0, 1))

            # Set state dimension
            self.state_dim = self.state.shape[-1]

        # Get data dimensions
        self.sequences, self.timesteps, self.d1, self.d2 = self.images.shape
        # We set controls to zero (we don't use them even if dim_u=1). If they are fixed to one instead (and dim_u=1)
        # the B matrix represents a bias.
        self.controls = np.zeros((self.sequences, self.timesteps, config.dim_u), dtype=np.float32)

    def shuffle(self, shuffle_images=False):
        permutation = np.random.permutation(self.sequences)
        self.state = self.state[permutation]
        self.controls = self.controls[permutation]

        if shuffle_images:
            self.images = self.images[permutation]
