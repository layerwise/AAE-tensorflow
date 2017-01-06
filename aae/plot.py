"""
Code adapted from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
"""

import itertools
import os

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np



def plot_latent_exploration(model, sess, outdir="./plots", **kwargs):
    """
    Plotting utility to explore the (low-dimensional!) manifold of latent space
    """
    feed_dict = dict()
    zs = kwargs.get("zs", None)
    color = kwargs.get("color", False)
    save = kwargs.get("save", True)
    if zs:
        feed_dict.update({model.zs :zs})
    latent_space = sess.run(model.latent_exploration_op, feed_dict=feed_dict)

    nx = latent_space.shape[0]/model.dataset.image_shape[0]
    ny = latent_space.shape[1]/model.dataset.image_shape[1]

    plt.figure(figsize=(nx / 2, ny / 2))
    if not color:
        plt.imshow(latent_space[:,:,0],
                   cmap="Greys",
                   aspect="auto",
                   extent=((-1.,1.) * 2))
    # no axes
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis("off")
    plt.tight_layout()

    if save:
        title = "{}_latent_round_{}_{}.png".format(
            model.datetime, model.counter, "name")
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")
    plt.close()


def plot_encoding(model, sess, title=None, name="encoding",
                  datasets=("test","validation"), range_=(-4.,4.),
                  save=True, outdir="./plots", **kwargs):
    """
    Plotting utility to encode dataset images in (low dimensional!) latent space
    """
    # TODO: check for 2d
    title = (title if title else name)
    for dataset_name in datasets:
        dataset = getattr(model.dataset, dataset_name)
        feed_dict = {model.x_in_test: dataset.images}
        encoding = sess.run(model.z_dist_info_, feed_dict=feed_dict)
        centers = encoding[model.latent_dist.dist_info_keys[0]]
        ys, xs = centers.T

        plt.figure()
        plt.title("round {}: {} in latent space".format(model.counter,
                                                        dataset_name))
        kwargs = {'alpha': 0.8}

        classes = set(dataset.labels)
        if classes:
            colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
            kwargs['c'] = [colormap[i] for i in dataset.labels]
            # make room for legend
            ax = plt.subplot(111)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            handles = [mpatches.Circle((0,0), label=class_, color=colormap[i])
                        for i, class_ in enumerate(classes)]
            ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
                      fancybox=True, loc='center left')

        plt.scatter(xs, ys, **kwargs)

        # map range_ to standard deviations of the target distribution
        stddev = model.target_dist.stddev
        adjusted_range = (stddev*range_[0], stddev*range_[1])

        if range_:
            plt.xlim(adjusted_range)
            plt.ylim(adjusted_range)

        if save:
            title = "{}_encoding_{}_round_{}.png".format(
                model.datetime, dataset_name, model.counter)
            plt.savefig(os.path.join(outdir, title), bbox_inches="tight")
        plt.close()



def plot_reconstruction(model, sess, dataset="validation", n=10, cols=None,
                        outlines=True, save=True, name="subset",
                        outdir="./plots"):
    """Util to plot subset of inputs and reconstructed outputs"""
    cols = (cols if cols else n)
    rows = 2 * int(np.ceil(n / cols)) # doubled b/c input & reconstruction

    plt.figure(figsize = (cols * 2, rows * 2))
    image_shape = model.dataset.image_shape

    dataset_name = dataset
    dataset = getattr(model.dataset, dataset_name)

    x_in = dataset.images
    n = min(n, x_in.shape[0])

    x_in = x_in[:n,:]
    x_reconstructed = sess.run(model.x_reconstructed,
                               feed_dict={model.x_in_test: x_in})

    def drawSubplot(x_, ax_):
        plt.imshow(x_.reshape(image_shape[:2]), cmap="Greys")
        if outlines:
            ax_.get_xaxis().set_visible(False)
            ax_.get_yaxis().set_visible(False)
        else:
            ax_.set_axis_off()

    for i, x in enumerate(x_in, 1):
        # display original
        ax = plt.subplot(rows, cols, i) # rows, cols, subplot numbered from 1
        drawSubplot(x, ax)

    for i, x in enumerate(x_reconstructed, 1):
        # display reconstruction
        ax = plt.subplot(rows, cols, i + cols * (rows / 2))
        drawSubplot(x, ax)

    if save:
        title = "{}_reconstruction_{}_round_{}.png".format(
                    model.datetime, dataset_name, model.counter)
        plt.savefig(os.path.join(outdir, title), bbox_inches="tight")
    plt.close()



