import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
import collections
import numpy as np

#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell.py


def tfbernoulli(shape, probof0, dtype=tf.float32):
  return tf.select(tf.random_uniform(shape) - probof0 > 0., tf.ones(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))


class _DropoutRNNCell(rnn_cell.RNNCell):

  def __init__(self, num_units, input_size, probofdrop_in, probofdrop_st, activation=tf.tanh):
    self._num_units = num_units
    self._activation = activation
    # ensure dropout probability is a float between 0 and 1
    self._probof1_in = 1.0 - float(probofdrop_in)
    self._probof1_st = 1.0 - float(probofdrop_st)
    assert(self._probof1_in <= 1.0 and self._probof1_in >= 0.0 and self._probof1_st <= 1.0 and self._probof1_st >= 0.0)
    # initialize dropout masks to all ones (keep everything)
    self.insz = input_size
    self.stsz = num_units
    bname = str(type(self).__name__)+"_dropmask"
    self._dropMaskInput = tf.placeholder(dtype=tf.float32, shape=[None,input_size], name=bname+"Input")
    self._dropMaskState = tf.placeholder(dtype=tf.float32, shape=[None, num_units], name=bname+"State")
    self._latest_mask_input = None
    self._latest_mask_state = None
    print("built _DropoutRNNCell with input_size "+str(input_size)+" and num_units "+str(num_units))

  def expectation_drop_mask(self, batch_size):
    self._latest_mask_input = np.ones((batch_size,self.insz)) * self._probof1_in
    self._latest_mask_state = np.ones((batch_size,self.stsz)) * self._probof1_st

  def random_drop_mask(self, batch_size):
    self._latest_mask_input = np.random.binomial(1, self._probof1_in, size=(batch_size,self.insz))
    self._latest_mask_state = np.random.binomial(1, self._probof1_st, size=(batch_size,self.stsz))

  def get_mask_feed(self):
    # merge with other fed variables
    if self._latest_mask_input is None or self._latest_mask_state is None:
        print("Must call a _drop_mask() function first (expectation_drop_mask or random_drop_mask)")
        assert(False)
    return {self._dropMaskInput: self._latest_mask_input, self._dropMaskState: self._latest_mask_state}

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units


class DropoutBasicRNNCell(_DropoutRNNCell):

  def __init__(self, *args, **kwargs):
    _DropoutRNNCell.__init__(self, *args, **kwargs)

  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = activation(W * input + U * state + B)."""

    with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      assert(self._dropMaskInput.get_shape()[1:] == inputs.get_shape()[1:])
      assert(self._dropMaskState.get_shape()[1:] == state.get_shape()[1:])
      dropin = tf.mul(self._dropMaskInput, inputs)
      dropst = tf.mul(self._dropMaskState, state)

      output = self._activation(rnn_cell._linear([dropin, dropst], self._num_units, True))

    return output, output


class DropoutGRUCell(_DropoutRNNCell):

  def __init__(self, *args, **kwargs):
    _DropoutRNNCell.__init__(self, *args, **kwargs)

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    
    with vs.variable_scope(scope or type(self).__name__):
      if self._dropMaskInput.get_shape()[1:] != inputs.get_shape()[1:]:
        print("error: "+str(self._dropMaskInput.get_shape()[1:])+" != "+str(inputs.get_shape()[1:]))
        assert(False)
      if self._dropMaskState.get_shape()[1:] != state.get_shape()[1:]:
        print("error: "+str(self._dropMaskState.get_shape()[1:])+" != "+str(state.get_shape()[1:]))
        assert(False)
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



