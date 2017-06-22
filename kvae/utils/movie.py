import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def save_frames(images, filename):
    num_sequences, n_steps, w, h = images.shape

    fig = plt.figure()
    im = plt.imshow(combine_multiple_img(images[:, 0]), cmap=plt.cm.get_cmap('Greys'), interpolation='none')
    plt.axis('image')

    def updatefig(*args):
        im.set_array(combine_multiple_img(images[:, args[0]]))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=500, frames=n_steps)

    # Either avconv or ffmpeg need to be installed in the system to produce the videos!
    try:
        writer = animation.writers['avconv']
    except KeyError:
        writer = animation.writers['ffmpeg']
    writer = writer(fps=3)
    ani.save(filename, writer=writer)
    plt.close(fig)


def save_true_generated_frames(true, generated, filename):
    num_sequences, n_steps, w, h = true.shape

    # Background is 0, foreground as 1
    true = np.copy(true[:16])
    true[true > 0.1] = 1

    # Set foreground be near 0.5
    generated = generated * .5

    # Background is 1, foreground is near 0.5
    generated = 1 - generated[:16, :n_steps]

    # Subtract true from generated so background is 1, true foreground is 0,
    # and generated foreground is around 0.5
    images = generated - true
    # images[images > 0.5] = 1.

    fig = plt.figure()
    im = plt.imshow(combine_multiple_img(images[:, 0]), cmap=plt.cm.get_cmap('gist_heat'),
                    interpolation='none', vmin=0, vmax=1)
    plt.axis('image')

    def updatefig(*args):
        im.set_array(combine_multiple_img(images[:, args[0]]))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=500, frames=n_steps)

    try:
        writer = animation.writers['avconv']
    except KeyError:
        writer = animation.writers['ffmpeg']
    writer = writer(fps=3)
    ani.save(filename, writer=writer)
    plt.close(fig)


def movie_to_frame(images):
    n_steps, w, h = images.shape
    colors = np.linspace(0.4, 1, n_steps)
    image = np.zeros((w, h))
    for i, color in zip(images, colors):
        image = np.clip(image + i * color, 0, color)
    return image


def save_movie_to_frame(images, filename, idx=0, cmap='Blues'):
    # Collect to single image
    image = movie_to_frame(images[idx])

    # Flip it
    # image = np.fliplr(image)
    # image = np.flipud(image)

    f = plt.figure(figsize=[12, 12])
    plt.imshow(image, cmap=plt.cm.get_cmap(cmap), interpolation='none', vmin=0, vmax=1)

    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close(f)


def save_movies_to_frame(images, filename, cmap='Blues'):
    # Binarize images
    # images[images > 0] = 1.

    # Grid images
    images = np.swapaxes(images, 1, 0)
    images = np.array([combine_multiple_img(image) for image in images])

    # Collect to single image
    image = movie_to_frame(images)

    f = plt.figure(figsize=[12, 12])
    plt.imshow(image, cmap=plt.cm.get_cmap(cmap), interpolation='none', vmin=0, vmax=1)
    plt.axis('image')
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=80)
    plt.close(f)


def save_frames_to_png(images, filepath):
    num_sequences, n_steps, w, h = images.shape

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    for i in range(n_steps):
        f = plt.figure(figsize=[12, 12])
        plt.imshow(images[0, i], cmap=plt.gray(), interpolation='none')
        plt.savefig(filepath + '/img_%d.png' % i, format='png', bbox_inches='tight', dpi=80)
        plt.close(f)


def combine_multiple_img(images, table_size=4, indices=None):

    if indices is None:
        indices = range(table_size**2)

    i = 0
    height = images[0].shape[0]
    width = images[0].shape[1]
    img_out = np.zeros((height * table_size, width * table_size))
    for x in range(table_size):
        for y in range(table_size):
            xa, xb = x * height, (x + 1) * height
            ya, yb = y * width, (y + 1) * width
            img_out[xa:xb, ya:yb] = images[indices[i]]
            i += 1

    return img_out


if __name__ == '__main__':
    filename = 'box'
    npzfile = np.load("../../data/%s.npz" %filename)
    images = npzfile['images']
    save_frames(images, 'training_sequence_%s.mp4' % filename)
    save_movies_to_frame(images, 'training_sequence_%s' % filename)
