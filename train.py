from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model
import cPickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/wikipediaaa',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=256,
                       help='size of RNN hidden state')
    parser.add_argument('--learn_input_embedding', type=bool, default=False,
                       help='Learn input embedding? If false, the one-hot representation is fed directly into the first RNN.')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of layers in the RNN')
    parser.add_argument('--dropout', type=float, default=0.25,
                       help='probability to drop a unit')
    parser.add_argument('--model', type=str, default='dropgru',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.99,
                       help='decay rate for rmsprop')                       
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    train(args)

def printargs(args):
    print("data_dir == "+str(args.data_dir))
    print("save_dir == "+str(args.save_dir))
    print("rnn_size == "+str(args.rnn_size))
    print("learn_input_embedding == "+str(args.learn_input_embedding))
    print("num_layers == "+str(args.num_layers))
    print("dropout == "+str(args.dropout))
    print("model == "+str(args.model))
    print("batch_size == "+str(args.batch_size))
    print("seq_length == "+str(args.seq_length))
    print("num_epochs == "+str(args.num_epochs))
    print("save_every == "+str(args.save_every))
    print("grad_clip == "+str(args.grad_clip))
    print("learning_rate == "+str(args.learning_rate))
    print("decay_rate == "+str(args.decay_rate))
    print("init_from == "+str(args.init_from))

def train(args):
    print("training on \'"+args.data_dir+"\'")
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size
    
    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        print("RELOADING FROM CHECKPOING")
        # check if all necessary files exist 
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl')) as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme
        
        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl')) as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagreee on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagreee on dictionary mappings!"

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    print("====================================")
    printargs(args)
    print("====================================")
    model = Model(args)

    def validateonce(expectationdropout=True):
        data_loader.reset_batch_pointers()
        state = model.resetstate(expectationdropout=expectationdropout)
        start = time.time()
        losses = []
        truths = []
        for b in range(data_loader.num_batches_te):
            x, y = data_loader.next_batch_te()
            # shapes of x and y are (batchsize, seqlength); each element is an integer from 0 to (vocabsize-1)
            feed = {model.input_data: x, model.targets: y, model.initial_state: state}
            feed = model.extrafeed(feed)
            state, probs = sess.run([model.final_state, model.probs], feed)
            y = y.flatten()
            for ii in range(y.size):
                losses.append(-np.log2(probs[ii,y[ii]]))
            truths.append(y)
        end = time.time()
        testtimeperbatch = (end-start) / float(data_loader.num_batches_te)
        return (np.array(losses), truths, testtimeperbatch)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            # train model
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointers()
            state = model.resetstate()
            for b in range(data_loader.num_batches_tr):
                start = time.time()
                x, y = data_loader.next_batch_tr()
                # shapes of x and y are (batchsize, seqlength); each element is an integer from 0 to (vocabsize-1)
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                feed = model.extrafeed(feed)
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                if b % 500 == 0:
                    state = model.resetstate()
                    print("reset state in the midst of a training epoch, at batch "+str(b+1)+"/"+str(data_loader.num_batches_tr))
                if (e * data_loader.num_batches_tr + b) % 100 == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.num_batches_tr + b,
                                args.num_epochs * data_loader.num_batches_tr,
                                e+1, train_loss, end - start))
                if (e * data_loader.num_batches_tr + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches_tr-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches_tr + b)
                    print("model saved to {}".format(checkpoint_path))
            # validate model
            if False and args.dropout > 1e-3 and (e % 4 == 0):
                testlosses = None
                ytruths = []
                testtimeperbatch = 0.0
                if e > 95:
                    niters = 1000
                else:
                    niters = 11 #29
                for kk in range(niters):
                    theselosses, thesetruths, thistimeperbatch = validateonce(expectationdropout=False)
                    if testlosses is None:
                        testlosses = theselosses
                    else:
                        testlosses = np.concatenate((testlosses,theselosses))
                    ytruths.append(thesetruths)
                    testtimeperbatch += (thistimeperbatch / float(niters))
                    #print("kk == "+str(kk+1)+"/"+str(niters))
                testloss = np.mean(np.array(testlosses))
                testlossstd = np.std(np.array(testlosses))
                if niters > 100:
                    with open('MClosses_'+str(e)+'.pkl', 'wb') as ff:
                        cPickle.dump(testlosses, ff)
                        cPickle.dump(ytruths, ff)
                suffix = ", estimated from "+str(niters)+" MC samples"
            else:
                theselosses, _, testtimeperbatch = validateonce(expectationdropout=True)
                testloss   = np.mean(theselosses)
                testlossstd = np.std(theselosses)
                suffix = ", MC expectation"
            
            print("At the end of epoch "+str(e+1)+", validation loss "+str(testloss)+" w/ stddev "+str(testlossstd)+" ("+str(testtimeperbatch)+" sec per batch)"+suffix)

if __name__ == '__main__':
    main()
