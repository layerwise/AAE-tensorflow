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
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import argparse
try:
    sys.path.append(os.environ["DATASETS"])
except KeyError:
    pass

import tensorflow as tf

from aae.datasets import MnistDataSet
from aae.distributions import MeanBernoulli, MeanGaussian, Gaussian, Deterministic
from aae.adversarial_autoencoder import AAE

DATASET = MnistDataSet()

IMG_DIM = DATASET.image_dim

LATENT_DIM = 2

LOG_DIR = "./logs/mnist"
METAGRAPH_DIR = "./ckt/mnist" #TODO
PLOTS_DIR = "./plots/mnist"

# output_dist: MeanBernoulli(IN_DIM) / MeanGaussian(IN_DIM, fix_std=True)
# latent_dist: Gaussian(LATENT_DIM) / Deterministic(LATENT_DIM)

MNIST_AAE_HYPERPARAMS = {
    "output_dist": MeanBernoulli(IMG_DIM),
    "latent_dist": Gaussian(LATENT_DIM),
    "target_dist": Gaussian(LATENT_DIM, fix_std=True, stddev=3.),
    "dataset": DATASET,
    "batch_size": 128,
    "max_epoch": 1000,
    "updates_per_epoch": 200, # None for passing through entire dataset
    "learning_rates": [1e-4, 1e-4, 1e-4],
    "regularization": "adversarial",
    "network_type": "convolutional"
}

MNIST_VAE_HYPERPARAMS = {
    "output_dist": MeanBernoulli(IMG_DIM),
    "latent_dist": Gaussian(LATENT_DIM),
    "target_dist": Gaussian(LATENT_DIM, fix_std=True),
    "dataset": DATASET,
    "batch_size": 128,
    "max_epoch": 1000,
    "updates_per_epoch": 200,
    "learning_rates": [1e-3, 2e-4, 0],
    "regularization": "variational",
    "network_type": "fully_connected"
}

MNIST_VAAE_HYPERPARAMS = {
    "output_dist": MeanBernoulli(IMG_DIM),
    "latent_dist": Gaussian(LATENT_DIM),
    "target_dist": Gaussian(LATENT_DIM, fix_std=True, stddev=3.),
    "dataset": DATASET,
    "batch_size": 128,
    "max_epoch": 1000,
    "updates_per_epoch": 200,
    "learning_rates": [1e-3, 1e-4, 1e-4],
    "regularization": "combined",
    "network_type": "convolutional"
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""Implementation to train an Adversarial Autoencoder or Variational Autoencoder on MNIST""")
    model_type_parser = parser.add_mutually_exclusive_group(required=True)
    model_type_parser.add_argument("--aae",
                                   action="store_true",
                                   help='Train an Adversarial Autoencoder')
    model_type_parser.add_argument("--vae",
                                   action="store_true",
                                   help='Train a Variational Autoencoder')
    model_type_parser.add_argument("--vaae",
                                   action="store_true",
                                   help="""Train an Autoencoder with combined variational and adversarial regularisation""")

    args = parser.parse_args()

    tf.reset_default_graph()

    if args.aae:
        model = AAE(**MNIST_AAE_HYPERPARAMS)
    elif args.vae:
        model = AAE(**MNIST_VAE_HYPERPARAMS)
    elif args.vaae:
        model = AAE(**MNIST_VAAE_HYPERPARAMS)

    model.train()

