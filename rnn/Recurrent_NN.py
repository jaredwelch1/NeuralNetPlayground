import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import glob
import numpy as np
import math
import os
import time
from datetime import datetime

# some values to be used in model creation 

SEQLEN = 30 # length of seq of characters fed into rnn
BATCHSIZE = 400 # seq / batch 
ALPHASIZE = 98 # number of possible values for a character (important for softmax layer and input sizing)
INTERNALSIZE = 512 # size of GRU cell internally 
NLAYERS = 3 # how many are we stacking 
learning_rate = 0.001
dropout_pkeep = 0.8 # probability of dropout

# Specification of the supported alphabet (subset of ASCII-7)
# 10 line feed LF
# 32-64 numbers and punctuation
# 65-90 upper-case letters
# 91-97 more punctuation
# 97-122 lower-case letters
# 123-126 more punctuation
def convert_from_alphabet(a):
    """Encode a character
    :param a: one character
    :return: the encoded value
    """
    if a == 9:
        return 1
    if a == 10:
        return 127 - 30  # LF
    elif 32 <= a <= 126:
        return a - 30
    else:
        return 0 # unknown

# encoded values:
# unknown = 0
# tab = 1
# space = 2
# all chars from 32 to 126 = c-30
# LF mapped to 127-30
def convert_to_alphabet(c, avoid_tab_and_lf=False):
    """Decode a code point
    :param c: code point
    :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
    :return: decoded character
    """
    if c == 1:
        return 32 if avoid_tab_and_lf else 9  # space instead of TAB
    if c == 127 - 30:
        return 92 if avoid_tab_and_lf else 10  # \ instead of LF
    if 32 <= c + 30 <= 126:
        return c + 30
    else:
        return 0  
    
# helper function for encoding data
def encode_text(s):
    """Encode a string.
    :param s: a text string
    :return: encoded list of code points
    """
    return list(map(lambda a: convert_from_alphabet(ord(a)), s))


def read_data_files(directory, validation=True):
        """read data files based on glob given
        :param directory: glob for data directory (example "shakespeare/*.txt")
        :param validation: if True, sets last file aside as validation data
        :return training_text, validation text, list of loaded file names"""
        train_txt = []
        bookranges = []
        file_list = glob.glob(directory, recursive=True)
        for f in file_list:
            with open(f, "r") as file_txt:
                print("Loading file " + f)
                start = len(train_txt)
                train_txt.extend(encode_text(file_txt.read()))
                end = len(train_txt)
                # give us name of file and where it is in the text array
                bookranges.append({"start": start, "end": end, "name": f.rsplit("/", 1)[-1]})
        if len(bookranges) == 0:
            sys.exit('No training data has been found. Aborting')
        
        # For validation, use roughly 90K of text,
        # but no more than 10% of the entire text
        # and no more than 1 book in 5 => no validation at all for 5 files or fewer.

        # 10% of the text is how many files ?
        total_len = len(train_txt)
        validation_len = 0
        nb_books1 = 0
        for book in reversed(bookranges):
            validation_len += book["end"]-book["start"]
            nb_books1 += 1
            if validation_len > total_len // 10:
                break

        # 90K of text is how many books ?
        validation_len = 0
        nb_books2 = 0
        for book in reversed(bookranges):
            validation_len += book["end"]-book["start"]
            nb_books2 += 1
            if validation_len > 90*1024:
                break

        # 20% of the books is how many books ?
        nb_books3 = len(bookranges) // 5

        # pick the smallest
        nb_books = min(nb_books1, nb_books2, nb_books3)

        if nb_books == 0 or not validation:
            cutoff = len(train_txt)
        else:
            cutoff = bookranges[-nb_books]["start"]
        valitext = train_txt[cutoff:]
        training_text = train_txt[:cutoff]
        return training_text, valitext, bookranges


# location of training/test text 
data_dir_glob = "shakespeare/*.txt"

# train_text and vali_text are lists of characters, encoded to numeric values to be fed to the rnn in batches
train_text, vali_text, file_names = read_data_files(data_dir_glob)


epoch_size = len(train_text) // (BATCHSIZE * SEQLEN)

# Need a special helper function to give a generator object that yields our batches to us
# This will ensure that the batches are properly setup so that sequences continue between batches

# example:
#     Batch 1 [ The cow jumped ove] Batch 2 [r the moon]
# 
# If we do not maintain the ordering of sequences in this manner, training is not as effective 
def rnn_minibatching(raw_data, batch_size, sequence_size, nb_epochs):
    data = np.array(raw_data)
    data_len = data.shape[0]
    # using (data_len - 1) so that we can make sure the y+1 sequence is also handled
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data for a single batch"
    rounded_data_len = int(nb_batches * batch_size * sequence_size)
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len+1], [batch_size, nb_batches * sequence_size])
    
    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch

# Create the model
lr = tf.placeholder(tf.float32, name='lr')
pkeep = tf.placeholder(tf.float32, name='pkeep')
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X') # will be [BATCHSIZE, SEQLEN]

# one-hot encode our input taking it from 2d to 3d tensor
Xo = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)              # will be [BATCHSIZE, SEQLEN, ALPHASIZE]

# do the same for output, which is same char seq shifted by 1 character
Y_ = tf.placeholder(tf.uint8, [None, None], name="Y_") # [BATCHSIZE, SEQLEN]
Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)              # [BATCHSIZE, SEQLEN, ALPHASIZE]

# Input state for our rnn (see documentation on RNN if not sure what this is compared to input data)
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name="Hin") # [BATCHSIZE, INTERNALSIZE*NLAYERS]

# NLAYERS=3 GRU cells unrolled over a SEQLEN of 30 chars
# dynamic_rnn can infer SEQLEN from input size

cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
# dropout implementation
dropcells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)

Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE]
# H: [ BATCHSIZE, INTERNALSIZE*NLAYERS ]

H = tf.identity(H, name='H') # give it a name

# Softmax output layer 
# Flatten the first two dimensions of the output [ BATCHSIZE, SEQLEN, ALPHASIZE] => [BATCHSIZE * SEQLEN, ALPHASIZE]
# then apply softmax readout

Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])   # [ BATCHSIZE x SEQLEN, INTERNALSIZE]
Ylogits = layers.linear(Yflat, ALPHASIZE)    # [ BATCHSIZE X SEQLEN, ALPHASIZE]
Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE])    # [ BATCHSIZE x SEQLEN, ALPHASIZE]
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Yflat_) # [ BATCHSIZE x SEQLEN ]
loss = tf.reshape(loss, [batchsize, -1])     # [ BATCHSIZE, SEQLEN ]
Yo = tf.nn.softmax(Ylogits, name="Yo")       # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Y = tf.argmax(Yo, 1)                         # [ BATCHSIZE x SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y") # [ BATCHSIZE, SEQLEN ]
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# init values and session

istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS]) # init state
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

# grab some statistics stuff to show during training

seqloss = tf.reduce_mean(loss, 1)
batchloss = tf.reduce_mean(seqloss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# init some tensorboard stuff. Saves tensor into folder of log files
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

# this will be a place to save trained models
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)

# setup some stuff to handle displaying progress
DISPLAY_FREQ = 50 # show every 100 batches
_100_batches = DISPLAY_FREQ * SEQLEN * BATCHSIZE


# Training 

NUM_EPOCHS = 10
prev_epoch = -1
epoch_start_time = datetime.now()
for x, y_, epoch in rnn_minibatching(train_text, BATCHSIZE, SEQLEN, nb_epochs=NUM_EPOCHS):
    if epoch != prev_epoch:
        if prev_epoch != -1:
            print("Training finished on epoch" + str(prev_epoch))
            curr_time = datetime.now()
            epoch_time = curr_time - epoch_start_time
            print("Time for training (hours:minutes:seconds)" + str(epoch_time))
            epoch_start_time = curr_time
            
        print("Training starting on epoch " + str(epoch))
        prev_epoch = epoch
    # train on a minibatch 
    feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
    _, y, ostate, = sess.run([train_step, Y, H], feed_dict=feed_dict)
    
    # log training data for tensorboard
    if step % _100_batches == 0:
        feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: BATCHSIZE}  # no dropout for validation
        y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
        summary_writer.add_summary(smm, step)
    
    # log a validation step 
    if step % _100_batches == 0 and len(vali_text) > 0:
        VALI_SEQLEN = 1*1024 # Sequence length for validation 
        bsize = len(vali_text) // VALI_SEQLEN
        vali_x, vali_y, _ = next(rnn_minibatching(vali_text, bsize, VALI_SEQLEN, 1)) # all data in a single batch
        vali_nullstate = np.zeros([bsize, INTERNALSIZE*NLAYERS])
        feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, pkeep: 1.0, batchsize: bsize}
        ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
        validation_writer.add_summary(smm, step)
    
    # save a checkpoint 
    if step // 10 % _100_batches == 0:
        saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
        print("Saved checkpoint file: " + saved_file)
    
    istate = ostate
    step += BATCHSIZE * SEQLEN
    

