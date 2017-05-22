import sys
import os
import numpy as np
import tensorflow as tf
from kvae.models import KalmanVariationalAutoencoder as Model
from kvae.utils import reload_config, get_image_config

import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})
np.random.seed(1337)


def run(model_dir):
    config = get_image_config().FLAGS
    config.log_dir = model_dir
    config.reload_model = os.path.join(model_dir, 'model.ckpt')
    config = reload_config(config)
    config.use_vae = False

    # Set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    with tf.Session() as sess:
        model = Model(config, sess)
        model.build_model().initialize_variables()
        model.generate()
        # model.impute(model.mask_impute_planning(4, 12), 4)
        model.impute(model.mask_impute_random(t_init_mask=4, drop_prob=0.8), 4)
        # model.imputation_plot('missing_planning')
        # model.imputation_plot('missing_random')

if __name__ == "__main__":
    run(sys.argv[1])
