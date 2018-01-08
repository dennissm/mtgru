import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell
      
class GRUCell(RNNCell):
    '''A Vanilla GRU Cell'''
    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h, h = state
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 2 * self.num_units],
                initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 2 * self.num_units],
                initializer=orthogonal_initializer())
            bias = tf.get_variable('bias', [2 * self.num_units])
            W_xu = tf.get_variable('W_xu',
                [self.num_units, 1 * self.num_units],
                initializer=orthogonal_initializer())
            W_hu = tf.get_variable('W_hu',
                [self.num_units, 1 * self.num_units],
                initializer=orthogonal_initializer())
            bias1 = tf.get_variable('bias1', [1 * self.num_units])
            
            concat = tf.concat(1, [x, h])
            W_both = tf.concat(0, [W_xh, W_hh])
            hidden = tf.matmul(concat, W_both) + bias

            z, r = tf.split(1, 2, hidden)
            z, r = tf.sigmoid(z), tf.sigmoid(r)
            
            u = tf.tanh((tf.matmul(W_xu, x) + tf.matmul(h * r, W_hu)) + bias1)
            
            new_h = (1 - z) * h + z * u

            return new_h, (new_h, new_h)



class MTGRUCell(RNNCell):
    '''An MTGRU Cell'''
    def __init__(self, num_units, tau):
        self.num_units = num_units
        self.tau = tau

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h, h = state
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                [x_size, 2 * self.num_units],
                initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                [self.num_units, 2 * self.num_units],
                initializer=orthogonal_initializer())
            bias = tf.get_variable('bias', [2 * self.num_units])
            W_xu = tf.get_variable('W_xu',
                [self.num_units, 1 * self.num_units],
                initializer=orthogonal_initializer())
            W_hu = tf.get_variable('W_hu',
                [self.num_units, 1 * self.num_units],
                initializer=orthogonal_initializer())
            bias1 = tf.get_variable('bias1', [1 * self.num_units])

            concat = tf.concat(1, [x, h])
            W_both = tf.concat(0, [W_xh, W_hh])
            hidden = tf.matmul(concat, W_both) + bias

            z, r = tf.split(1, 2, hidden)
            z, r = tf.sigmoid(z), tf.sigmoid(r)
            
            u = tf.tanh((tf.matmul(W_xu, x) + tf.matmul(h * r, W_hu)) + bias1)
            
            new_h = z * h + (1 - z) * u

            new_h = (1 - self.tau) * h + self.tau * new_h

            return new_h, (new_h, new_h)


def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer

