# Copyright 2015 Mathias Schmerling
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# =============================================================================
"""
Tensorflow implementation of an Adversarial Autoencoder (https://arxiv.org/abs/1511.05644)
"""

import os
import sys
import numpy as np
import errno
from datetime import datetime

import tensorflow as tf
import prettytensor as pt
from progressbar import ETA, Bar, Percentage, ProgressBar

from aae.distributions import Deterministic, Gaussian, Bernoulli
from aae.utils import mkdir_p
import aae.plot as plot
import aae.custom_ops

TINY = 1e-8
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5

class AAE():
    """Adversarial Autoencoders

    see: Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, Ian Goodfellow, Brendan Frey
    (https://arxiv.org/abs/1511.05644)

    Central class for training an adversarial autoencoder.

    Attributes:
        output_dist:
            instance of a distribution class, e.g. MeanBernoulli
        latent_dist:
            instance of a distribution class, e.g. Gaussian
        target_dist:
            instance of a distribution class, e.g. Gaussian
        dataset:
            a dataset class with attributes 'image_dim' and 'image_shape' and
            a method 'next_batch'.
        network_type: {'fully-connected', 'convolutional'},
        batch_size: int
        max_epoch: int
        updates_per_epoch: int or None
        learning_rates: list(3)
            a list of up to three entries for the reconstruction, generator
            and discriminator trainer
        regularization: {'adversarial', 'variational', 'combined'}
        exp_name: str
        log_dir: str
        plot_dir: str
        checkpoint_interval: int
        counter: int

    Methods:
        encode
        decode
        generate
        autoencode
        reconstruct
        train
        visualize
    """
    def __init__(self,
                 output_dist,
                 latent_dist,
                 target_dist,
                 dataset,
                 network_type="convolutional",
                 batch_size=128,
                 max_epoch=100,
                 updates_per_epoch=None,
                 learning_rates=[1e-3, 2e-4, 2e-4],
                 regularization="adversarial",
                 exp_name="experiment",
                 log_dir="logs",
                 ckt_dir="ckt",
                 plot_dir="plots",
                 checkpoint_interval=10000):
        """Initialize AAE class

        Args:
            output_dist: Distribution
                instance of a distribution class. Determines loss function. I.E.
                MeanBernoulli(in_dim) for binary cross entropy loss and
                MeanGaussian(in_dim, fix_std=True) for quadratic error loss
            latent_dist: Distribution
                instance of a distribution class, e.g. Gaussian(latent_dim) or
                Deterministic(latent_dim). The former will invoke the
                reparametrization trick of Kingma & Welling (2014)
            target_dist: Distribution
                determines the latent distribution that the encoder/generator is
                trying to match (via adversarial training or kl-loss)
            dataset:
                a dataset class with attributes 'image_dim' and 'image_shape' and
                a method 'next_batch'.
            network_type: {'fully-connected', 'convolutional'}, optional
                default 'convolutional'
            batch_size: int, optional
                default 128
            max_epoch: int, optional
                default 100
            updates_per_epoch: int or None, optional
                Default None for passing through entire dataset in one epoch
            learning_rates: list(3), optional
                a list of up to three entries for the reconstruction, generator
                and discriminator trainer. Default [1e-3, 2e-4, 2e-4]
            regularization: {'adversarial', 'variational', 'combined'}, optional
                determines the type of regularization, default 'adversarial'
            exp_name: str, optional
            log_dir: str, optional
            plot_dir: str, optional
            checkpoint_interval: int, optional
        """

        self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")

        self.output_dist = output_dist
        self.latent_dist = latent_dist
        self.target_dist = target_dist
        self.dataset = dataset
        self.network_type = network_type
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        if updates_per_epoch:
            self.updates_per_epoch = updates_per_epoch
        else:
            self.updates_per_epoch = (self.dataset.train.num_examples
                                      / self.batch_size)
        self.learning_rates = learning_rates
        self.regularization = regularization

        self.image_shape = dataset.image_shape
        self.image_dim = dataset.image_dim

        self.exp_name = exp_name
        self.log_dir = os.path.join(log_dir, self.datetime+"_"+exp_name)
        self.ckt_dir = os.path.join(ckt_dir, self.datetime+"_"+exp_name)
        self.plot_dir = os.path.join(plot_dir, self.datetime+"_"+exp_name)
        self.checkpoint_interval = checkpoint_interval

        mkdir_p(self.log_dir)
        mkdir_p(self.plot_dir)
        mkdir_p(self.ckt_dir)

        self.log_vars = []

        self.counter = 0

        self._build_graph()

        # plotting
        self.plotting_steps = []
        BASE = 2
        INCREMENT = 1
        pow_ = 0
        while BASE**pow_ < self.max_epoch*self.updates_per_epoch:
            self.plotting_steps.append(BASE**pow_)
            pow_ += INCREMENT
        self.plotting_steps.append(self.max_epoch*self.updates_per_epoch)
        self.plot_latent = True
        self.plot_encoding = True
        self.plot_reconstruction = True

    def _make_encoder_template(self):
        defaults_scope = {
            'phase': pt.UnboundVariable('phase', default=pt.Phase.train),
            'scale_after_normalization': True,
            }
        with pt.defaults_scope(**defaults_scope):
          with tf.variable_scope("encoder"):
            if self.network_type=="fully-connected":
                z_dim = self.latent_dist.dist_flat_dim
                self.encoder_template = (pt.template("x_in").
                                         custom_fully_connected(1000).
                                         batch_normalize().
                                         apply(tf.nn.elu).
                                         custom_fully_connected(1000).
                                         batch_normalize().
                                         apply(tf.nn.elu).
                                         custom_fully_connected(z_dim))

            elif self.network_type=="convolutional":
                z_dim = self.latent_dist.dist_flat_dim
                self.encoder_template = (pt.template("x_in").
                                         reshape([-1] + list(self.image_shape)).
                                         custom_conv2d(64, k_h=4, k_w=4).
                                         apply(tf.nn.elu).
                                         custom_conv2d(128, k_h=4, k_w=4).
                                         batch_normalize().
                                         apply(tf.nn.elu).
                                         custom_fully_connected(1024).
                                         batch_normalize().
                                         apply(tf.nn.elu).
                                         custom_fully_connected(z_dim))

    def _make_decoder_template(self):
        defaults_scope = {
            'phase': pt.UnboundVariable('phase', default=pt.Phase.train),
            'scale_after_normalization': True,
            }
        image_size = self.image_shape[0]
        with pt.defaults_scope(**defaults_scope):
          with tf.variable_scope("decoder"):
            if self.network_type=="fully-connected":
                self.decoder_template = (pt.template("z_in").
                                         custom_fully_connected(1000).
                                         apply(tf.nn.relu).
                                         custom_fully_connected(1000).
                                         batch_normalize().
                                         apply(tf.nn.relu).
                                         custom_fully_connected(self.image_dim))

            elif self.network_type=="convolutional":
                self.decoder_template = \
                    (pt.template("z_in").
                     custom_fully_connected(1024).
                     batch_normalize().
                     apply(tf.nn.relu).
                     custom_fully_connected(image_size/4 * image_size/4 * 128).
                     batch_normalize().
                     apply(tf.nn.relu).
                     reshape([-1, image_size/4, image_size/4, 128]).
                     custom_deconv2d([0, image_size/2, image_size/2, 64],
                                     k_h=4, k_w=4).
                     batch_normalize().
                     apply(tf.nn.relu).
                     custom_deconv2d([0] + list(self.image_shape),
                                     k_h=4, k_w=4).
                     flatten())

    def _make_discriminator_template(self):
        defaults_scope = {
            'phase': pt.UnboundVariable('phase', default=pt.Phase.train),
            'scale_after_normalization': True,
            }
        with pt.defaults_scope(**defaults_scope):
          with tf.variable_scope("discriminator"):
            self.discriminator_template = (pt.template("z_in").
                                           custom_fully_connected(1000).
                                           apply(tf.nn.relu).
                                           custom_fully_connected(1000).
                                           batch_normalize().
                                           apply(tf.nn.relu).
                                           custom_fully_connected(1))

    def _build_graph(self):
        """Build the computational graph.

        Placeholders:
            x_in: shape(None, dataset.image_dim)
                training input images
            x_in_test: shape(None, dataset.image_dim)
                input to the testing graph (different because of batch norm)
            z_prior_
                for direct exploration of latent space

        Training ops:
            h_encoded:
                encoder output (pre-activation)
            z:
                latent space output, sampled from self.latent_dist
            h_decoded
                decoder output (pre-activation)
            rec_loss
            kl_loss
            gen_loss
            dis_loss
            reconst_trainer
            generator_trainer
            discriminator_trainer

        Testing and visualization ops:
            h_encoded_:
                encoder output for test graph
            z_dist_info_: dict
                distribution parameters for latent distribution
            z_:
                sample from latent distribution
            h_decoded_;
                decoder output for test graph
            x_reconstructed:
                reconstructed x_in_test
            h_decoded_prior_:
                decoder output for direct exploration of latent space
            x_generated:
                generated image for direct_exploration of latent space
            latent_exploration_op:
                op for a canvas of generated images along the percentiles of
                the latent dimension (2d only)
        """
        self.x_in = tf.placeholder(tf.float32,
                                   shape=[None, self.image_dim],
                                   name="x_in")

        self._make_encoder_template()
        self._make_decoder_template()
        self._make_discriminator_template()

        self.log_vars = []

        # ----------------------------- -------- ----------------------------- #
        # ----------------------------- Training ----------------------------- #
        # ----------------------------- -------- ----------------------------- #

        # compute encoder output
        self.h_encoded = self.encoder_template.construct(x_in=self.x_in).tensor

        # map encoder output to distribution parameters
        z_dist_info = self.latent_dist.activate_dist(self.h_encoded)

        # sample from latent distribution using encoded parameters
        # (will invoke the reparametrization trick e.g. for Gaussian dist)
        self.z = self.latent_dist.sample(z_dist_info)

        # compute decoder output
        self.h_decoded = self.decoder_template.construct(z_in=self.z).tensor

        # map decoder output to the parameters of the output distribution
        # (e.g. sigmoid squashing nonlinearity for Bernoulli output)
        x_dist_info = self.output_dist.activate_dist(self.h_decoded)

        # compute cross-entropy loss between input and output distribution
        # (equivalent to negative log-likelihood)
        # and average over minibatch
        self.rec_loss = tf.reduce_mean( \
                            - self.output_dist.logli(self.x_in,
                                                     x_dist_info))

        # logging
        self.log_vars.append(("reconstruction_loss", self.rec_loss))

        if self.regularization in ["adversarial","combined"]:

            # sample from target prior distribution
            self.z_prior = self.target_dist.sample_prior(self.batch_size)

            # discriminate negative and positive samples
            real_d = self._discriminate(self.z_prior)
            fake_d = self._discriminate(self.z)

            # compute generator and discriminator loss
            self.gen_loss = -tf.reduce_mean(tf.log(fake_d + TINY))
            self.dis_loss = -tf.reduce_mean(tf.log(real_d + TINY)
                                            + tf.log(1. - fake_d + TINY))

            # logging variables
            self.log_vars.append(("generator_loss", self.gen_loss))
            self.log_vars.append(("discriminator_loss", self.dis_loss))

            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))

        if self.regularization in ["variational","combined"]:

            # assumes a Gaussian posterior and prior
            self.kl_loss = tf.reduce_mean( \
                              self.target_dist.kl_prior(z_dist_info,
                                                        self.batch_size))

            # logging variables
            self.log_vars.append(("kl_loss", self.kl_loss))

        # build train ops
        all_vars = tf.trainable_variables()
        enc_vars = [var for var in all_vars if var.name.startswith('encoder')]
        dec_vars = [var for var in all_vars if var.name.startswith('decoder')]
        dis_vars = [var for var in all_vars if var.name.startswith('discriminator')]

        r_vars = enc_vars + dec_vars
        d_vars = dis_vars
        g_vars = enc_vars

        # if variational or combined regularization: add kl loss
        if self.regularization in ["variational","combined"]:
            reconst_optimizer = \
                tf.train.AdamOptimizer(self.learning_rates[0], beta1=0.5)
            r_grads_and_vars = \
                reconst_optimizer.compute_gradients(self.rec_loss
                                                    + self.kl_loss,
                                                    r_vars)

        # else only reconstruction loss
        elif self.regularization=="adversarial":
            reconst_optimizer = \
                tf.train.AdamOptimizer(self.learning_rates[0], beta1=0.5)
            r_grads_and_vars = \
                reconst_optimizer.compute_gradients(self.rec_loss,
                                                    r_vars)

        r_clipped = [(tf.clip_by_norm(r_grad, 10.), r_var)
                      for r_grad, r_var in r_grads_and_vars]
        self.reconst_trainer = reconst_optimizer.apply_gradients(r_clipped)


        # if adversarial or combined regularization: adversarial trainers
        if self.regularization in ["adversarial","combined"]:
            generator_optimizer = \
                tf.train.AdamOptimizer(self.learning_rates[1], beta1=0.5)
            g_grads_and_vars = \
                generator_optimizer.compute_gradients(self.gen_loss,
                                                      g_vars)
            g_clipped = [(tf.clip_by_norm(g_grad, 10.), g_var)
                          for g_grad, g_var in g_grads_and_vars]
            self.generator_trainer = (generator_optimizer.
                                      apply_gradients(g_clipped))

            discriminator_optimizer = \
                tf.train.AdamOptimizer(self.learning_rates[2], beta1=0.5)
            d_grads_and_vars = \
                discriminator_optimizer.compute_gradients(self.dis_loss,
                                                          d_vars)
            d_clipped = [(tf.clip_by_norm(d_grad, 10.), d_var)
                          for d_grad, d_var in d_grads_and_vars]
            self.discriminator_trainer = (discriminator_optimizer.
                                          apply_gradients(d_clipped))

        #self.global_step = tf.Variable(0, trainable=False) TODO

        # --------------------- ------------------------- -------------------- #
        # --------------------- Testing and Visualization -------------------- #
        # --------------------- ------------------------- -------------------- #
        # test phase encoding of input
        self.x_in_test = tf.placeholder(tf.float32,
                                        shape=[None, self.image_dim],
                                        name="x_in_test")
        h_encoded_ = self.encoder_template.construct(x_in=self.x_in_test,
                                                     phase=pt.Phase.test)
        self.z_dist_info_ = self.latent_dist.activate_dist(h_encoded_)

        self.z_ = self.latent_dist.sample(self.z_dist_info_)

        # test phase decoding of latent space
        h_decoded_ = self.decoder_template.construct(z_in=self.z_,
                                                     phase=pt.Phase.test)
        x_dist_info_ = self.output_dist.activate_dist(h_decoded_)
        self.x_reconstructed = self.output_dist.sample(x_dist_info_)

        self.test_rec_loss = tf.reduce_mean( \
                                - self.output_dist.logli(self.x_in_test,
                                                         x_dist_info_))

        # enable direct exploration of latent space
        self.z_prior_ = tf.placeholder_with_default( \
                            self.target_dist.sample_prior(1),
                            shape=[None, self.target_dist.dim],
                            name="z_in")

        h_decoded_prior_ = self.decoder_template.construct(z_in=self.z_prior_,
                                                           phase=pt.Phase.test)
        x_prior_dist_info = self.output_dist.activate_dist(h_decoded_prior_)
        self.x_generated = self.output_dist.sample(x_prior_dist_info)

        try:
            self._make_latent_exploration_op()
        except NotImplementedError:
            self.latent_exploration_op = None

        # cannot be part of log vars because not evaluated at each update
        #self.log_vars.append(("validation_loss", self.test_rec_loss))

        # logging
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)
        tf.summary.scalar("validation_loss", self.test_rec_loss)

    def _make_latent_exploration_op(self):
        """
        Code adapted from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
        """
        # ops for exploration of latent space
        nx = 30
        ny = nx
        z_dim = self.target_dist.dim
        
        if self.latent_dist.dim==2:
            range_ = (0, 1)
            min_, max_ = range_
            zs = np.rollaxis(np.mgrid[max_:min_:ny*1j, min_:max_:nx*1j], 0, 3)
            if isinstance(self.target_dist, Gaussian):
                from scipy.stats import norm
                DELTA = 1E-8 # delta to avoid +/- inf at 0, 1 boundaries
                zs = np.array([norm.ppf(np.clip(z, TINY, 1 - TINY),
                                        scale=self.target_dist.stddev)
                               for z in zs])
        else:
            raise NotImplementedError

        zs = tf.constant(zs.reshape((nx*ny, z_dim)),
                         dtype=tf.float32)

        self.zs = tf.placeholder_with_default(zs,
                                              shape=[None, z_dim],
                                              name="zs")

        hs_decoded = self.decoder_template.construct(z_in=self.zs,
                                                     phase=pt.Phase.test).tensor
        xs_dist_info = self.output_dist.activate_dist(hs_decoded)
        xs = self.output_dist.sample(xs_dist_info)

        imgs = tf.reshape(xs, [nx, ny] + list(self.dataset.image_shape))
        stacked_img = []
        for row in xrange(nx):
            row_img = []
            for col in xrange(ny):
                row_img.append(imgs[row, col, :, :, :])
            stacked_img.append(tf.concat(axis=1, values=row_img))
        self.latent_exploration_op = tf.concat(axis=0, values=stacked_img)

    def _discriminate(self, z_var):
        d_out = self.discriminator_template.construct(z_in=z_var)
        d = tf.nn.sigmoid(d_out)
        return d

    def encode(self, x, sess=None):
        """Inference q(z|x): Probabilistically encodes the input x to the latent
        distribution parameters.
        """
        if sess is not None:
            sess = tf.Session()
        feed_dict = {self.x_in: x}
        return sess.run(self.z_dist_info, feed_dict=feed_dict)

    def decode(self, zs=None, sess=None):
        """Generation p(x|z): Reconstruct images by decoding the latent space"""
        if sess is not None:
            sess = tf.Session()
        if zs is not None:
            feed_dict = {self.z_prior_: zs}
        else:
            feed_dict = dict()
        return sess.run(self.x_generated, feed_dict=feed_dict)

    def generate(self, sess=None):
        """Generates an image"""
        return self.decode(sess=sess)

    def autoencode(self, x, sess=None):
        """End-to-end autoencoder. Will return the reconstruction of x"""
        if sess is not None:
            sess = tf.Session()
        feed_dict = {self.x_in_test: x}
        return sess.run(self.x_reconstructed, feed_dict=feed_dict)

    def reconstruct(self, x, sess=None):
        """Same as AAE.autoencode"""
        return self.autoencode(x, sess=sess)

    def train(self):
        """Starts the training"""

        init = tf.global_variables_initializer() #tf.global_variables_initializer()

        with tf.Session(config=config) as sess:
            sess.run(init)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            saver = tf.train.Saver()

            log_vars = [x for _, x in self.log_vars]
            log_keys = [x for x, _ in self.log_vars]

            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()

                all_log_vals = []
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    x, _ = self.dataset.train.next_batch(self.batch_size)
                    feed_dict = {self.x_in: x}

                    # --- reconstruction phase --- #
                    log_vals = sess.run([self.reconst_trainer] + log_vars,
                                        feed_dict)[1:]

                    all_log_vals.append(log_vals)

                    # --- regularization phase --- #
                    if self.regularization in ["adversarial","combined"]:
                        # discriminator
                        sess.run(self.discriminator_trainer, feed_dict)
                        # generator
                        sess.run(self.generator_trainer, feed_dict)

                    self.counter += 1

                    if self.counter in self.plotting_steps:
                        self.visualize(sess)

                    if self.counter % self.checkpoint_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(self.counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.ckt_dir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                x, _ = self.dataset.train.next_batch(self.batch_size)
                x_valid = self.dataset.validation.images

                summary_str = sess.run(summary_op,
                                       {self.x_in: x,
                                        self.x_in_test: x_valid})
                summary_writer.add_summary(summary_str, self.counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_dict = dict(zip(log_keys, avg_log_vals))

                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                print("Epoch %d | " % (epoch) + log_line)
                sys.stdout.flush()
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")
            summary_writer.flush()
            summary_writer.close()

    def visualize(self, sess=None, **kwargs):
        """Visualize the training process. Will plot the latent space canvas,
        the encoding of samples in latent space and the reconstruction for
        some sample images.

        args:
            sess: tensorflow.Session, optional
            kwargs: 
                plotting kwargs, transmitted to low-level plotting
                facilities
        """
        if not sess:
            sess = tf.Session()
        if self.plot_latent:
            plot.plot_latent_exploration(self, sess, outdir=self.plot_dir,
                                         **kwargs)

        if self.plot_encoding:
            plot.plot_encoding(self, sess, outdir=self.plot_dir,**kwargs)

        if self.plot_reconstruction:
            plot.plot_reconstruction(self, sess, outdir=self.plot_dir,**kwargs)


