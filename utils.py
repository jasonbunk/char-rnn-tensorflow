import codecs
import os
import collections
from six.moves import cPickle
import numpy as np

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointers()

    def preprocess(self, input_file, vocab_file, tensor_file):
        # first pass: scan file and count unique characters
        chunksz = int(1e8)
        counter = collections.Counter()
        nbytes = 0
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            keeplooping = True
            while keeplooping:
                thisdata = f.read(size=chunksz)
                thislen = len(thisdata)
                nbytes += thislen
                if thislen < chunksz:
                    keeplooping = False
                if thislen > 0:
                    counter.update(thisdata)
                print("counted chars in "+str(float(nbytes)/1e6)+" megabytes")
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        with open(vocab_file+'.txt','w') as f:
            for char in self.chars:
                f.write('count '+str(counter[char])+' -- code '+str(ord(char))+' '+str(char)+'\n')
        print("While preprocessing \'"+input_file+"\', chose vocabulary size "+str(self.vocab_size)+"... characters:")
        print(str(self.chars))
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        # second pass: convert characters to integer representation mapped by character index
        self.tensor = None
        nbytes = 0
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            keeplooping = True
            while keeplooping:
                thisdata = f.read(size=chunksz)
                thislen = len(thisdata)
                nbytes += thislen
                if thislen < chunksz:
                    keeplooping = False
                if thislen > 0:
                    thisnparr = np.array(list(map(self.vocab.get, thisdata)), dtype=np.uint8)
                    if self.tensor is None:
                        self.tensor = thisnparr
                    else:
                        self.tensor = np.concatenate((self.tensor,thisnparr))
                print("converted "+str(float(nbytes)/1e6)+" megabytes (self.tensor.shape == "+str(self.tensor.shape)+")")
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        print("From \'"+vocab_file+"\', vocabulary size "+str(self.vocab_size)+"... characters:")
        print(str(self.chars))
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)

    def create_batches(self):
        num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

        # When the data (tesor) is too small, let's give them a better error message
        if num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:(num_batches * self.batch_size * self.seq_length)]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        x_batches = np.split(xdata.reshape(self.batch_size, -1), num_batches, 1)
        y_batches = np.split(ydata.reshape(self.batch_size, -1), num_batches, 1)
        assert(len(x_batches) == len(y_batches))
        randombatchidxs = np.random.permutation(len(x_batches))
        numtestbatches = int(round(float(len(x_batches))*0.1)) # 10% for validation, 90% for training
        self.x_batches_te = x_batches[:numtestbatches]
        self.y_batches_te = y_batches[:numtestbatches]
        self.x_batches_tr = x_batches[numtestbatches:]
        self.y_batches_tr = y_batches[numtestbatches:]
        self.num_batches_te = len(self.x_batches_te)
        self.num_batches_tr = len(self.x_batches_tr)
        if self.num_batches_te == 0 or self.num_batches_tr == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        print("Set up with "+str(self.num_batches_tr)+" training batches and "+str(self.num_batches_te)+" validation batches")

    def next_batch_tr(self):
        x, y = self.x_batches_tr[self.pointer_tr], self.y_batches_tr[self.pointer_tr]
        self.pointer_tr = (self.pointer_tr + 1) % self.num_batches_tr
        return x, y

    def next_batch_te(self):
        x, y = self.x_batches_te[self.pointer_te], self.y_batches_te[self.pointer_te]
        self.pointer_te = (self.pointer_te + 1) % self.num_batches_te
        return x, y

    def reset_batch_pointers(self):
        self.pointer_tr = 0
        self.pointer_te = 0
