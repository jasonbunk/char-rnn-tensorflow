from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf

import argparse
import time
from six.moves import cPickle

from utils import TextLoader
from model import Model

from six import text_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=2000,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u'\n',
                       help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Softmax sampling temperature, must be greater than zero; for very small values << 1, becomes equivalent to argmax() at each sampling step; for very large values >> 1, becomes equivalent to uniformly random selection')

    args = parser.parse_args()
    sample(args)

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("=================================================")
            print(model.sample(sess, chars, vocab, args.n, args.prime, args.sample, temperature=args.temperature))
            print("=================================================")

if __name__ == '__main__':
    main()
