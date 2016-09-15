import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.util import nest
from dropgru import DropoutGRUCell, DropoutBasicRNNCell

import numpy as np

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            self.batch_size = 1
            self.seq_length = 1
        else:
            self.batch_size = args.batch_size
            self.seq_length = args.seq_length

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        elif args.model == 'dropgru' or args.model == 'droprnn':
            pass
        else:
            raise Exception("model type not supported: {}".format(args.model))

        if args.model.startswith('drop'):
            cells = []
            
            dt1 = DropoutBasicRNNCell
            dt2 = DropoutGRUCell
            if args.model != 'dropgru':
                print("additional layers will be basic RNN")
                dt2 = DropoutBasicRNNCell
            
            for ii in range(args.num_layers):
                if False and args.learn_input_embedding:
                    # context-dependent embedding learned as a small RNN before the large GRUs
                    args.learn_input_embedding = False
                    if ii == 0:
                        nc = dt1(args.vocab_size, input_size=args.vocab_size, probofdrop_st=args.dropout, probofdrop_in=0.0)
                    elif ii == 1:
                        nc = dt2(args.rnn_size,   input_size=args.vocab_size, probofdrop_st=args.dropout, probofdrop_in=args.dropout)
                    else:
                        nc = dt2(args.rnn_size,   input_size=args.rnn_size,   probofdrop_st=args.dropout, probofdrop_in=args.dropout)
                else:
                    # embedding is fixed, context-independent; like word vectors
                    firstdroprate = 0.0
                    if args.learn_input_embedding:
                        firstdroprate = args.dropout
                    if ii == 0:
                        nc = dt2(args.rnn_size, input_size=args.vocab_size, probofdrop_st=args.dropout, probofdrop_in=firstdroprate)
                    else:
                        nc = dt2(args.rnn_size, input_size=args.rnn_size,   probofdrop_st=args.dropout, probofdrop_in=args.dropout)
                cells.append(nc)
            
            self.cell = rnn_cell.MultiRNNCell(cells)
            self.cellusesdropout = True
        else:
            print("building basic non-dropout model")
            c1 = cell_fn(args.rnn_size)
            self.cell = rnn_cell.MultiRNNCell([c1] * args.num_layers)
            self.cellusesdropout = False

        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.seq_length], name="x_input_data")
        self.targets = tf.placeholder(tf.int32, [self.batch_size, self.seq_length], name="y_targets")
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        if args.learn_input_embedding:
            self.embedding = tf.get_variable("embedding", [args.vocab_size, args.vocab_size])
        else:
            self.embedding = tf.placeholder(tf.float32, [args.vocab_size, args.vocab_size], name="embedding")

        if self.cellusesdropout:
            self._dropMaskOutput = tf.placeholder(dtype=tf.float32, shape=[self.batch_size*self.seq_length, args.rnn_size], name="dropout_output_mask")
            self._latest_mask_output = None

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("top_softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("top_softmax_b", [args.vocab_size])
            inputs = tf.split(1, self.seq_length, tf.nn.embedding_lookup(self.embedding, self.input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            if self.cellusesdropout:
                assert(prev.get_shape() == self._dropMaskOutput.get_shape())
                prev = tf.matmul(tf.mul(prev, self._dropMaskOutput), softmax_w) + softmax_b
            else:
                prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(self.embedding, prev_symbol)

        self.temperature = tf.placeholder(tf.float32, 1, name="temperature")

        # if loop_function is not None, it is used to generate the next input
        # otherwise, if it is None, the next input will be from the "inputs" sequence
        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, self.cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [self.batch_size*self.seq_length, args.rnn_size])

        if self.cellusesdropout:
            assert(output.get_shape() == self._dropMaskOutput.get_shape())
            self.logits = tf.matmul(tf.mul(output, self._dropMaskOutput), softmax_w) + softmax_b
        else:
            self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        self.probswithtemp = tf.nn.softmax(self.logits / self.temperature)

        # 1.44... term converts cost from units of "nats" to units of "bits"
        self.cost = seq2seq.sequence_loss([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([self.batch_size * self.seq_length])]) * 1.44269504088896340736
        self.pred_entropy = tf.reduce_sum(tf.mul(self.probs, tf.log(self.probs + 1e-12)), 1) * (-1.44269504088896340736)

        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False, name="learningrate")
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        zipgradvars = zip(grads, tvars)
        self.train_op = optimizer.apply_gradients(zipgradvars)

        # for tensorboard
        tb_cost = tf.scalar_summary('cost_train', self.cost)
        tb_predent = tf.scalar_summary('prediction_entropy_train', tf.reduce_mean(self.pred_entropy))
        mergethese = [tb_cost, tb_predent]
        for grad,var in zipgradvars:
            mergethese.append(tf.histogram_summary(var.name+'_value', var))
            mergethese.append(tf.histogram_summary(var.name+'_grad', grad))
        self.tbsummary = tf.merge_summary(mergethese)

    def extrafeed(self, feed):
        if self.args.learn_input_embedding == False:
            feed[self.embedding] = np.identity(self.args.vocab_size, dtype=np.float32)
        if self.cellusesdropout:
            for cell in self.cell._cells:
                feed.update(cell.get_mask_feed())
            assert(self._latest_mask_output is not None)
            feed[self._dropMaskOutput] = self._latest_mask_output
        return feed

    def resetstate(self):
        if nest.is_sequence(self.initial_state):
            if nest.is_sequence(self.initial_state[0]):
                state = tuple(tuple(is2.eval() for is2 in ist) for ist in self.initial_state)
            else:
                state = tuple(ist.eval() for ist in self.initial_state)
        else:
            state = self.initial_state.eval()
        return state

    def resetweights(self, expectationdropout=False):
        if self.cellusesdropout:
            for cell in self.cell._cells:
                if expectationdropout:
                    cell.expectation_drop_mask(self.batch_size)
                else:
                    cell.random_drop_mask(self.batch_size)
        outdropsize = (self.batch_size*self.seq_length, self.args.rnn_size)
        if self.cellusesdropout:
            if expectationdropout:
                self._latest_mask_output = np.ones(outdropsize) * (1.0 - float(self.args.dropout))
            else:
                self._latest_mask_output = np.random.binomial(1, (1.0 - float(self.args.dropout)), size=outdropsize)

    def sample(self, sess, chars, vocab, num=200, prime=' ', sampling_type=1, temperature=1.0):
        self.resetweights(expectationdropout=True)
        state = self.resetstate()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            feed = self.extrafeed(feed)
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state, self.temperature: (float(temperature),)}
            feed = self.extrafeed(feed)
            [probs, state] = sess.run([self.probswithtemp, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret


