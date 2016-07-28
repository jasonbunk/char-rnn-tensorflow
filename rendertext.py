from __future__ import print_function
import os, cPickle
import numpy as np
import tensorflow as tf

def writestr(char, intensity):
    codes = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'cx']
    assert(intensity >= 0 and intensity < len(codes))
    ic = codes[intensity]
    charwasnewline = False
    if char == '\n':
        char = '<fg>\\n</fg>'
        charwasnewline = True
    retstr = '<'+ic+'>'+char+'</'+ic+'>'
    if charwasnewline:
        return retstr + '<br>'
    return retstr

def rendertext(headercolor, save_dir, fname, chardata, intensitydata):
    headertxt = ''
    with open(headercolor+'header.txt','r') as headerfile:
        for line in headerfile:
            headertxt += line
    if len(chardata.shape) != 2 or len(intensitydata.shape) != 2:
        print("rendertext -- inputs must be of dimension 2, shape (batch-idx, seq-idx)")
        assert(False)
    if chardata.shape != intensitydata.shape:
        print("rendertext -- inputs must be of same shape")
        assert(False)
    if intensitydata.dtype != np.float32 and intensitydata.dtype != np.float64:
        print("rendertext -- intensitydata must have dtype float32 or float64")
        assert(False)
    
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    
    with open(os.path.join(save_dir, fname+'.html'), 'w') as outfile:
        outfile.write(headertxt)
        for ii in range(min(50,chardata.shape[0])):
            if ii > 0:
                outfile.write('\n\n<br><br>===============================<br><br>\n\n')
            themax = np.amax(intensitydata[ii,:])
            ktable = [np.sqrt(float(kk+1)/11.0)*themax for kk in range(11)]
            for jj in range(chardata.shape[1]):
                for kk in range(11):
                    if kk == 10 or intensitydata[ii,jj] <= ktable[kk]:
                        outfile.write(writestr(chars[chardata[ii,jj]], kk))
                        break
                    #elif kk == 10:
                    #    print("intensitydata["+str(ii)+","+str(jj)+"] == "+str(intensitydata[ii,jj])+", max is "+str(themax)+", binwidth is "+str(binwidth))
                    #    assert(False)
        outfile.write('\n\n</p></body></html>')
