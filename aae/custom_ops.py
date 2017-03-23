"""
Code taken from https://github.com/openai/InfoGAN/blob/master/infogan/misc/custom_ops.py
and slightly modified.
"""

import prettytensor as pt
import tensorflow as tf
from prettytensor.pretty_tensor_class import Phase
import numpy as np


@pt.Register
class custom_conv2d(pt.VarStoreMethod):
    def __call__(self, input_layer, output_dim,
                 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, in_dim=None, padding='SAME',
                 name="conv2d"):
        with tf.variable_scope(name):
            w = self.variable('w', [k_h, k_w, in_dim or input_layer.shape[-1], output_dim],
                              init=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_layer.tensor, w, strides=[1, d_h, d_w, 1], padding=padding)

            biases = self.variable('biases', [output_dim], init=tf.constant_initializer(0.0))
            # import ipdb; ipdb.set_trace()
            return input_layer.with_tensor(tf.nn.bias_add(conv, biases), parameters=self.vars)


@pt.Register
class custom_deconv2d(pt.VarStoreMethod):
    def __call__(self, input_layer, output_shape,
                 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                 name="deconv2d"):
        # code modification: allow for variable input to enable building the
        # graph with unknown batch size, from https://github.com/tensorflow/tensorflow/issues/833
        dyn_input_shape = tf.shape(input_layer)
        batch_size = dyn_input_shape[0]
        ts_output_shape = tf.stack([batch_size,
                                   output_shape[1],
                                   output_shape[2],
                                   output_shape[3]])
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = self.variable('w', [k_h, k_w, output_shape[-1], input_layer.shape[-1]],
                              init=tf.random_normal_initializer(stddev=stddev))

            try:
                deconv = tf.nn.conv2d_transpose(input_layer, w,
                                                output_shape=ts_output_shape,
                                                strides=[1, d_h, d_w, 1])

            # Support for versions of TensorFlow before 0.7.0
            except AttributeError:
                deconv = tf.nn.deconv2d(input_layer, w, output_shape=ts_output_shape,
                                        strides=[1, d_h, d_w, 1])

            biases = self.variable('biases', [output_shape[-1]], init=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), [-1] + output_shape[1:])

            return deconv


@pt.Register
class custom_fully_connected(pt.VarStoreMethod):
    def __call__(self, input_layer, output_size, scope=None, in_dim=None, stddev=0.02, bias_start=0.0):
        shape = input_layer.shape
        input_ = input_layer.tensor
        if True:#try:
            if len(shape) == 4:
                input_ = tf.reshape(input_, tf.stack([tf.shape(input_)[0], np.prod(shape[1:])]))
                input_.set_shape([None, np.prod(shape[1:])])
                shape = input_.get_shape().as_list()

            with tf.variable_scope(scope or "Linear"):
                matrix = self.variable("Matrix", [in_dim or shape[1], output_size], dt=tf.float32,
                                       init=tf.random_normal_initializer(stddev=stddev))
                bias = self.variable("bias", [output_size], init=tf.constant_initializer(bias_start))
                return input_layer.with_tensor(tf.matmul(input_, matrix) + bias, parameters=self.vars)
        #except Exception:
        #    import ipdb; ipdb.set_trace()
