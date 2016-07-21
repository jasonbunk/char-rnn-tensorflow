import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
import collections
import numpy as np

#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py


def tfbernoulli(shape, probof0, dtype=tf.float32):
  return tf.select(tf.random_uniform(shape) - probof0 > 0., tf.ones(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))


class DropoutGRUCell(rnn_cell.RNNCell):

  def __init__(self, num_units, batch_size, input_size, probofdrop, activation=tf.tanh):
    self._num_units = num_units
    self._activation = activation
    # ensure dropout probability is a float between 0 and 1
    self._probof1 = 1.0 - float(probofdrop)
    assert(self._probof1 <= 1.0 and self._probof1 >= 0.0)
    # initialize dropout masks to all ones (keep everything)
    self.insh = (batch_size, input_size)
    self.stsh = (batch_size, num_units)
    bname = str(type(self).__name__)+"_dropmask"
    self._dropMaskInput = tf.placeholder(dtype=tf.float32, shape=[None,input_size], name=bname+"Input")
    self._dropMaskState = tf.placeholder(dtype=tf.float32, shape=[None, num_units], name=bname+"State")
    self._latest_mask_input = np.ones(self.insh, dtype=np.float32)
    self._latest_mask_state = np.ones(self.stsh, dtype=np.float32)

  def reset_drop_mask(self):
    self._latest_mask_input = np.random.binomial(1, self._probof1, size=self.insh) / self._probof1
    self._latest_mask_state = np.random.binomial(1, self._probof1, size=self.stsh) / self._probof1

  def get_mask_feed(self):
    # merge with other fed variables
    return {self._dropMaskInput: self._latest_mask_input, self._dropMaskState: self._latest_mask_state}

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    
    with vs.variable_scope(scope or type(self).__name__):
      assert(self._dropMaskInput.get_shape()[1:] == inputs.get_shape()[1:])
      assert(self._dropMaskState.get_shape()[1:] == state.get_shape()[1:])
      
      dropin = tf.mul(self._dropMaskInput, inputs)
      dropst = tf.mul(self._dropMaskState, state)

      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        concat = rnn_cell._linear([dropin, dropst], 2 * self._num_units, True, 1.0)
        r, u = tf.split(1, 2, concat)
        r, u = tf.sigmoid(r), tf.sigmoid(u)

      with vs.variable_scope("Candidate"):
        htilda = self._activation(rnn_cell._linear([dropin, r * dropst], self._num_units, True))

      new_h = u * dropst + (1 - u) * htilda

    return new_h, new_h









