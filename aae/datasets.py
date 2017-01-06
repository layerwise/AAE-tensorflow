
import os
import sys
import numpy as np
from tensorflow.examples.tutorials import mnist

try:
    DATA_DIRECTORY = os.path.join(os.environ["DATASETS"])
except KeyError:
    DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "..") 



class DataSet(object):
    def __init__(self, images, supervised, flatten=False,
                 shuffle_initially=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        #dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        self.dtype = dtype
        assert len(images) == len(supervised), \
                ('images.shape: %s labels.shape: %s' % (images.shape,
                                                        supervised.shape))

        self._image_dim = np.product(images.shape[-3:-1]) 
        self._image_shape = images.shape[-3:]

        self._num_examples = images.shape[0]
        self._images = images
        self._supervised = supervised
        if flatten:
            self._images = self._images \
                              .reshape(self._images.shape[0],
                                       np.product(self._images.shape[1:]))
        if self.dtype == np.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            self._images = self._images.astype(np.float32)
            self._images = np.multiply(self._images, 1.0 / 255.0)
        if shuffle_initially:
            self.shuffle()
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        # legacy
        return self._supervised
    @property
    def supervised(self):
        return self._supervised
    @property
    def episodic(self):
        return self._episodic
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def shuffle(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._supervised = self._supervised[perm]
        self._episode_indices = self._episode_indices[perm]
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self.shuffle()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end,...], self._supervised[start:end,...]



# ---------------------------------- ----- ----------------------------------- #
# ---------------------------------- MNIST ----------------------------------- #
# ---------------------------------- ----- ----------------------------------- #
class MnistDataSet(object):
    """
    TODO: add source - infogan
    """
    def __init__(self):
        self.data_directory = os.path.join(DATA_DIRECTORY,
                                           "mnist")
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        dataset = mnist.input_data.read_data_sets(self.data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        # wtf is this? TODO
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = DataSet(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


