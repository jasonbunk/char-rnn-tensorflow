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

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
plt.ion()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/wikipediaaa',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=192,
                       help='size of RNN hidden state')
    parser.add_argument('--learn_input_embedding', type=bool, default=False,
                       help='Learn input embedding? If false, the one-hot representation is fed directly into the first RNN.')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of layers in the RNN')
    parser.add_argument('--dropout', type=float, default=0.33333333,
                       help='probability to drop a unit')
    parser.add_argument('--model', type=str, default='dropgru',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=500,
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

def tryconcat(arr1, arr2, axis):
    if arr1 is None:
        return arr2
    return np.concatenate((arr1,arr2),axis=axis)

def plothist(data, xlabel, titlestr):
    data = data.flatten()
    mu = np.mean(data)
    sigma = np.std(data)
    n, bins, patches = plt.hist(data, 100, normed=1, facecolor='green', alpha=0.75)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{'+titlestr+':}\ \mu='+str(mu)+',\ \sigma='+str(sigma)+'$')
    plt.grid(True)

def entropyvariance(args, MCprobs, MCentrs, plotfig=None):
    assert(len(MCprobs.shape) == 4)
    assert(MCprobs.shape[0] == 1) # should have already been averaged
    assert(MCprobs.shape[1] == args.batch_size)
    assert(MCprobs.shape[2] % args.seq_length == 0)
    assert(MCprobs.shape[3] == args.vocab_size)
    assert(len(MCentrs.shape) == 3)
    assert(MCentrs.shape == MCprobs.shape[:3])
    totalseqlen = MCprobs.shape[2]
    # shape must be: (1, batch_size, seq_length*num_seqs, vocab_size)
    MCprobs = np.reshape(MCprobs,(MCprobs.shape[1]*MCprobs.shape[2], MCprobs.shape[3]))
    MCentrs = np.reshape(MCentrs,(MCentrs.shape[1]*MCentrs.shape[2]))
    # sum across vocab_size
    entrofmeans = -1.0 * np.sum(MCprobs * np.log2(MCprobs+1e-12), axis=1)
    # compute JS Divergences
    JSdivergences = entrofmeans - MCentrs
    # reshape
    JSdivergences = np.reshape(JSdivergences, (args.batch_size, totalseqlen))
    # plot
    if plotfig is not None:
        plt.figure(plotfig)
        plt.clf()
        # skip first 10 which may be more unpredictable (state is "warming up")
        plothist(JSdivergences[:, 10:], 'JS Divergence', 'JS Divergences of MC outputs')
        plt.draw()
    return JSdivergences

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
        entrps = None
        truths = None
        allprobs = None
        for b in range(data_loader.num_batches_te):
            x, y = data_loader.next_batch_te()
            # shapes of x and y are (batchsize, seqlength); each element is an integer from 0 to (vocabsize-1)
            feed = {model.input_data: x, model.targets: y, model.initial_state: state}
            feed = model.extrafeed(feed)
            state, probs, entropies = sess.run([model.final_state, model.probs, model.pred_entropy], feed)
            theseprobs = np.reshape(probs, (1, args.batch_size, args.seq_length, args.vocab_size))
            thesey = np.reshape(y, (args.batch_size, args.seq_length))
            allprobs = tryconcat(allprobs, theseprobs, axis=2)
            truths = tryconcat(truths, thesey, axis=1)
            y = y.flatten()
            for ii in range(y.size):
                losses.append(-np.log2(probs[ii,y[ii]]))
            thesentropies = np.reshape(entropies,(1,args.batch_size,args.seq_length))
            entrps = tryconcat(entrps, thesentropies, axis=2)
        end = time.time()
        testtimeperbatch = (end-start) / float(data_loader.num_batches_te)
        return (np.array(losses), truths, entrps, allprobs, testtimeperbatch)

    # for tensorboard
    valsumplh_cost = tf.placeholder(tf.float32, (1,), name="validation_summary_placeholder_cost")
    valsumplh_pent = tf.placeholder(tf.float32, (1,), name="validation_summary_placeholder_prediction_entropy")
    #reduce_sum fixes tensorflow scalar handling being weird (vector of size 1)
    valsumscs_cost = tf.scalar_summary('cost_val', tf.reduce_sum(valsumplh_cost))
    valsumscs_pent = tf.scalar_summary('prediction_entropy_val', tf.reduce_sum(valsumplh_pent))
    sumwriter = tf.train.SummaryWriter(args.save_dir, graph=tf.get_default_graph())

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        print(" ")
        allvars = tf.all_variables()
        trainablevars = tf.trainable_variables()
        for tvar in allvars:
            #print(type(tvar))
            #print(tvar.name+" -- "+str(tvar.dtype)+" -- "+str(tvar.get_shape()))
            if tvar in trainablevars:
                print("@@@ "+tvar.name+" -- "+str(tvar.get_shape()))
            else:
                print(tvar.name+" -- "+str(tvar.get_shape()))
        print(" ")

        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            # train model
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointers()
            state = model.resetstate()
            for b in range(data_loader.num_batches_tr):
                dovalidate = False
                if b == (data_loader.num_batches_tr - 1):
                    dovalidate = True
                start = time.time()
                x, y = data_loader.next_batch_tr()
                # shapes of x and y are (batchsize, seqlength); each element is an integer from 0 to (vocabsize-1)
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                feed = model.extrafeed(feed)
                train_loss, state, _, summary = sess.run([model.cost, model.final_state, model.train_op, model.tbsummary], feed)
                end = time.time()
                bidx = e * data_loader.num_batches_tr + b
                sumwriter.add_summary(summary, bidx)
                epstr = "{}/{} (epoch {})".format(bidx, args.num_epochs * data_loader.num_batches_tr, e+1)
                if bidx % 100 == 0:
                    print(epstr + ", train_loss = {:.3f}, time/batch = {:.3f}".format(train_loss, end - start))
                if bidx % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches_tr-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = bidx)
                    print(epstr+", model saved to {}".format(checkpoint_path))
                if b > 0 and b % 500 == 0:
                    state = model.resetstate()
                    if b % 2500 == 0:
                        dovalidate = True
                    print(epstr+", reset state in the midst of a training epoch, at batch "+str(b+1)+"/"+str(data_loader.num_batches_tr))
                # validate model?
                if dovalidate:
                    befvaltime = time.time()
                    if args.dropout > 1e-3:
                        testlosses = None
                        ytruths = None
                        meanpredentrops = None
                        meanprobdistrs = None
                        testtimeperbatch = 0.0
                        if e > 95:
                            niters = 29
                        else:
                            niters = 5
                        for kk in range(niters):
                            theselosses, thesetruths, theseentrops, theseprobs, thistimeperbatch = validateonce(expectationdropout=False)
                            testlosses = tryconcat(testlosses, theselosses, axis=0)
                            if meanprobdistrs is None:
                                meanprobdistrs = theseprobs
                                meanpredentrops = theseentrops
                            else:
                                meanprobdistrs += theseprobs
                                meanpredentrops += theseentrops
                            if ytruths is None:
                                ytruths = thesetruths
                            testtimeperbatch += (thistimeperbatch / float(niters))
                        meanprobdistrs /= float(niters)
                        meanpredentrops /= float(niters)
                        entropvar = entropyvariance(args, meanprobdistrs, meanpredentrops, plotfig=1)
                        testloss = np.mean(testlosses)
                        testlossstd = np.std(testlosses)
                        if niters > 100:
                            with open('MClosses_'+str(e)+'.pkl', 'wb') as ff:
                                cPickle.dump(testlosses, ff)
                                cPickle.dump(ytruths, ff)
                        valpredentropy = np.mean(meanpredentrops)
                        valpredentrstd = np.std( meanpredentrops)
                        suffix = ", estimated from "+str(niters)+" MC samples"
                    else:
                        theselosses, _, theseentrops, _, testtimeperbatch = validateonce(expectationdropout=True)
                        testloss   = np.mean(theselosses)
                        testlossstd = np.std(theselosses)
                        valpredentropy = np.mean(theseentrops)
                        valpredentrstd = np.std( theseentrops)
                        suffix = ", MC expectation"
                    
                    valsummary1 = sess.run([valsumscs_cost,], {valsumplh_cost:np.array(testloss).reshape((1,))})[0]
                    valsummary2 = sess.run([valsumscs_pent,], {valsumplh_pent:np.array(valpredentropy).reshape((1,))})[0]
                    sumwriter.add_summary(valsummary1, (e+1)*data_loader.num_batches_tr)
                    sumwriter.add_summary(valsummary2, (e+1)*data_loader.num_batches_tr)
                    
                    aftvaltime = time.time()
                    
                    print(epstr+", val loss "+str(testloss)+" w/std "+str(testlossstd)+", pred-ent "+str(valpredentropy)+" w/std "+str(valpredentrstd)+" ("+str(testtimeperbatch)+" sec per batch)"+suffix)
                    print("validation time: "+str(aftvaltime-befvaltime)+" sec")

if __name__ == '__main__':
    main()
