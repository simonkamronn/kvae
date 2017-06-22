import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})

import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib import cm
from kvae.utils.movie import movie_to_frame
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.spatial.distance import hamming

matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14


def plot_auxiliary(all_vars, filename, table_size=4):
    # All variables need to be (batch_size, sequence_length, dimension)
    for i, a in enumerate(all_vars):
        if a.ndim == 2:
            all_vars[i] = np.expand_dims(a, 0)

    dim = all_vars[0].shape[-1]
    if dim == 2:
        f, ax = plt.subplots(table_size, table_size, sharex='col', sharey='row', figsize=[12, 12])
        idx = 0
        for x in range(table_size):
            for y in range(table_size):
                for a in all_vars:
                    # Loop over the batch dimension
                    ax[x, y].plot(a[idx, :, 0], a[idx, :, 1], linestyle='-', marker='o', markersize=3)
                    # Plot starting point of the trajectory
                    ax[x, y].plot(a[idx, 0, 0], a[idx, 0, 1], 'r.', ms=12)
                idx += 1
        # plt.show()
        plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
        plt.close()
    else:
        df_list = []
        for i, a in enumerate(all_vars):
            df = pd.DataFrame(all_vars[i].reshape(-1, dim))
            df['class'] = i
            df_list.append(df)

        df_all = pd.concat(df_list)
        sns_plot = sns.pairplot(df_all, hue="class", vars=range(dim))
        sns_plot.savefig(filename)
    plt.close()


def plot_alpha(alpha, filename, idx=0):
    fig = plt.figure(figsize=[6, 6])
    ax = fig.gca()

    for line in np.swapaxes(alpha[idx], 1, 0):
        ax.plot(line, linestyle='-')

    ax.set_xlabel('Steps', fontsize=30)
    ax.set_ylabel('Mixture weight', fontsize=30)
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close()


def plot_alpha_grid(alpha, filename, table_size=4, idx=0):
    # All variables need to be (batch_size, sequence_length, dimension)
    if alpha.ndim == 2:
        alpha = np.expand_dims(alpha, 0)

    f, ax = plt.subplots(table_size, table_size, sharex='col', sharey='row', figsize=[12, 12])
    for x in range(table_size):
        for y in range(table_size):
            for i in range(alpha.shape[-1]):
                ax[x, y].plot(alpha[idx, :, i], linestyle='-', marker='o', markersize=3)
                ax[x, y].set_ylim([-0.01, 1.01])
            idx += 1
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close()


def construct_ball_trajectory(var, r=1., cmap='Blues', start_color=0.4, shape='c'):
    # https://matplotlib.org/examples/color/colormaps_reference.html
    patches = []
    for pos in var:
        if shape == 'c':
            patches.append(mpatches.Circle(pos, r))
        elif shape == 'r':
            patches.append(mpatches.RegularPolygon(pos, 4, r))
        elif shape == 's':
            patches.append(mpatches.RegularPolygon(pos, 6, r))

    colors = np.linspace(start_color, .9, len(patches))
    collection = PatchCollection(patches, cmap=cm.get_cmap(cmap), alpha=1.)
    collection.set_array(np.array(colors))
    collection.set_clim(0, 1)
    return collection


def plot_ball_trajectory(var, filename, idx=0, scale=30, cmap='Blues'):
    # Calc optimal radius of ball
    x_min, y_min = np.min(var[:, :, :2], axis=(0, 1))
    x_max, y_max = np.max(var[:, :, :2], axis=(0, 1))
    r = max((x_max - x_min), (y_max - y_min)) / scale

    fig = plt.figure(figsize=[4, 4])
    ax = fig.gca()
    collection = construct_ball_trajectory(var[idx], r=1, cmap=cmap)
    ax.add_collection(collection)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")
    ax.set_xlabel('$a_{t,1}$', fontsize=24)
    ax.set_ylabel('$a_{t,2}$', fontsize=24)

    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close()


def plot_ball_trajectories(all_vars, filename, table_size=4, scale=30):
    """ Plot trajectory with balls
    :param all_vars: batch size x sequence length x dimensions
    :param filename: path and filename to save to
    :param table_size: grid size
    :return: None
    """

    # Calc optimal radius of ball
    x_min, y_min = np.min(all_vars[:, :, :2], axis=(0, 1))
    x_max, y_max = np.max(all_vars[:, :, :2], axis=(0, 1))
    r = (x_max - x_min) / scale

    fig, axes = plt.subplots(table_size, table_size, sharex=True, sharey=True, figsize=[12, 12])
    for idx, ax in enumerate(axes.flat):
        collection = construct_ball_trajectory(all_vars[idx], r=r)
        ax.axis("equal")
        ax.add_collection(collection)

    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close()


def plot_ball_trajectories_comparison(enc, gen, impute, filename, idx=0, scale=60, nrows=1, ncols=3, mask=None):

    if isinstance(idx, int):
        idx = np.arange(nrows*ncols)

    # Calc optimal radius of ball
    x_min, y_min = np.min(enc, axis=(0, 1))
    x_max, y_max = np.max(enc, axis=(0, 1))
    r = (x_max - x_min) / scale

    fig, axes = plt.subplots(nrows, ncols, figsize=[ncols*6, nrows*6])
    for i, ax in enumerate(axes.flat):
        for var, cmap, c in zip([enc, gen, impute],
                                ['Reds', 'Blues', 'Greens'],
                                ['red', 'blue', 'green']):
            ax.plot(var[idx[i], :, 0], var[idx[i], :, 1], color=c, alpha=1, linewidth=2)
            collection = construct_ball_trajectory(var[idx[i]], r=r, cmap=cmap)
            ax.add_collection(collection)

            # if cmap == 'Reds':
            #     if mask is not None:
            #         collection = construct_ball_trajectory(enc[idx[i], mask[idx[i]] == 1], r * 1.7, cmap='Reds',
            #                                                start_color=.4, shape='r')
            #         ax.add_collection(collection)

        # Add the observed samples in the end
        collection = construct_ball_trajectory(enc[idx[i], mask[idx[i]] == 1], r * 1.5, cmap='Greys',
                                               start_color=.9, shape='r')
        ax.add_collection(collection)

        # Add the starting point
        collection = construct_ball_trajectory(enc[idx[i], [0]], r * 2, cmap='Greys',
                                               start_color=.9, shape='s')
        ax.add_collection(collection)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("equal")

        if i >= ncols*(nrows - 1):
            ax.set_xlabel('$a_{t,1}$', fontsize=30)
        if i % ncols == 0:
            ax.set_ylabel('$a_{t,2}$', fontsize=30)

        # ax.set_xlim(x_min - 1, x_max + 1)
        # ax.set_ylim(y_min - 1, y_max + 1)

    axes[0, 0].legend(['Encoded', 'Generated', 'Smoothed'], fontsize=30, loc=0)
    plt.tight_layout()
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close()


def plot_3d_ball_trajectory(var, filename, r=0.05):
    var = np.asarray(var)

    # Normalize directions
    var -= var.min(axis=0)
    var /= var.max(axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z in var:
        p = mpatches.Circle((x, y), r, ec="none")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")

        p = mpatches.Circle((x, z), r, ec="none")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="y")

        p = mpatches.Circle((y, z), r, ec="none")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")

        # ax.scatter(x, y, z, s=100)
    # ax.plot(var[:, 0], var[:, 1], zs=var[:, 2])

    ax.view_init(azim=45, elev=30)
    ax.set_xlim3d(-0.1, 1.1)
    ax.set_ylim3d(-0.1, 1.1)
    ax.set_zlim3d(-0.1, 1.1)
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close(fig)
    # plt.show()


def plot_trajectory_and_video(trajectory, images, filename, idx=5, cmap='Blues', sidebyside=True):
    # Create 2D trajectory
    collection = construct_ball_trajectory(trajectory[idx, :20], 1, cmap=cmap)

    # Create constructed images
    # images[images > 0] = 1.
    image = movie_to_frame(images[idx, :20])

    # Reverse y-axis in image
    # image = np.flipud(image)
    image = np.fliplr(image)

    if sidebyside:
        fig, ax = plt.subplots(ncols=2, figsize=[20, 10])

        x_min, y_min = np.min(trajectory, axis=(0, 1))
        x_max, y_max = np.max(trajectory, axis=(0, 1))

        ax[0].axis("equal")
        ax[0].add_collection(collection)
        ax[0].set_xlim([x_min, x_max])
        ax[0].set_ylim([y_min, y_max])
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(image, cmap=plt.cm.get_cmap(cmap), interpolation='none', vmin=0, vmax=1)
        ax[1].set_xlim([1, 31])
        ax[1].set_ylim([1, 31])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    else:
        fig, ax = plt.subplots(ncols=1, figsize=[12, 12])
        ax.add_collection(collection)
        ax.imshow(image, cmap=plt.cm.get_cmap('Reds'), interpolation='none', vmin=0, vmax=1)
        ax.set_xlim([1, 31])
        ax.set_ylim([1, 31])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.tick_params(bottom=False, left=False)
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close(fig)


def plot_ball_and_alpha(alpha, trajectory, filename, cmap='Blues'):
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=[12, 6])
    collection = construct_ball_trajectory(trajectory, r=1., cmap=cmap)

    x_min, y_min = np.min(trajectory, axis=0)
    x_max, y_max = np.max(trajectory, axis=0)

    ax[0].add_collection(collection)
    ax[0].set_xlim([x_min, x_max])
    ax[0].set_ylim([y_min, y_max])
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    ax[0].axis("equal")

    for line in np.swapaxes(alpha, 1, 0):
        ax[1].plot(line, linestyle='-')

    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close()


def plot_trajectory_uncertainty(true, gen, filter, smooth, filename):
    sequences, timesteps, h, w = true.shape

    errors = dict(Generated=list(), Filtered=list(), Smoothed=list())
    for label, var in zip(('Generated', 'Filtered', 'Smoothed'), (gen, filter, smooth)):
        for step in range(timesteps):
            errors[label].append(hamming(true[:, step].ravel() > 0.5, var[:, step].ravel() > 0.5))

        plt.plot(np.linspace(1, timesteps, num=timesteps).astype(int), errors[label], linewidth=3, ms=20, label=label)
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('Hamming distance', fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(filename)
    plt.close()


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


def plot_kalman_transfers(matrices, filename):
    fig, axarr = plt.subplots(1, len(matrices))
    for idx, mat in enumerate(matrices):
        hinton(mat, ax=axarr[idx])
    fig.savefig(filename, format='png', bbox_inches='tight', dpi=80)


if __name__ == '__main__':
    # hinton(np.random.rand(20, 20) - 0.5)
    # plt.show()
    # filename = 'box_rnd'
    # npzfile = np.load("../../data/%s.npz" %filename)
    # states = npzfile['state'][:, :, :2]
    # plot_auxiliary([states], 'plot_true_%s.png' %filename)
    # save_frames_to_png(images, 'training_sequence_img')

    filename = 'box_rnd'
    npzfile = np.load("../../data/%s.npz" %filename)
    state = npzfile['state']
    images = npzfile['images']

    # plot_ball_trajectories(state, 'trajectory_grid')
    # plot_trajectory_and_video(state, images, 'training_combined', sidebyside=True)

    # plot_3d_ball_trajectory(np.concatenate((state[0, :, :2], np.random.rand(state.shape[1], 1)*32), 1), '3d_plot')

    # mask = np.random.choice([0, 1], size=(16, 20), p=[0.5, 0.5])
    # plot_ball_trajectories_comparison(state[:16, :, :2], state[16:32, :, :2], state[32:48, :, :2], 'training_ball_comp',
    #                                   mask=mask)

    # plot_trajectory_uncertainty(images[:16, :], images[16:32, :], images[32:48, :], images[48:64, :],
    #                             'training_uncertainty')
    plot_ball_and_alpha(None, state[0, :, :2], '')
