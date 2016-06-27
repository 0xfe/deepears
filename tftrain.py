#!/usr/bin/env python
#
# $ source tf/bin/activate
# $ ./tftrain.py

import datetime
import sys

import tensorflow as tf
import scipy, numpy as np
from scipy.io import wavfile

from sklearn import metrics
from sklearn.metrics import confusion_matrix

# This import ordering is important for matplotlib and seaborn
# in virtualenv. Otherwise it looks for an interactive display which
# isn't available.
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns

N = 256  # FFT size
IMG_X = 200  # this can be changed (max 229 for .330ms sample)
IMG_Y = (N/2) + 1
NUM_INPUTS = IMG_X * IMG_Y # Total number of pixels
NUM_CLASSES = 12 # 12-tones

# Validation
TEST_SIZE=200

# Train
BATCH_SIZE=128
LEARNING_RATE=0.1

# One-hot encode values in batch.
def one_hot(batch, max_value):
    b = np.zeros((len(batch), max_value), dtype='int32')
    b[np.arange(len(batch)), batch] = 1
    return b

def decode_single(filename):
    # Note that num_epochs here is None, so it'll just start from the
    # top again when the input is done. Effectively allows you to
    # loop forever.
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=1)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([], tf.int64),
            'sample': tf.FixedLenFeature([NUM_INPUTS], tf.float32)
        })
    # now return the converted data
    label = features['label']
    sample = features['sample']
    return label, sample


def input_data():
    return samples_batch, labels_batch

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,      
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,      
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


X = tf.placeholder("float", [None, IMG_X, IMG_Y, 1])
Y = tf.placeholder("float", [None, NUM_CLASSES])

w = init_weights([16, 16, 1, 64])    
w2 = init_weights([16, 16, 64, 128])
w3 = init_weights([16, 16, 128, 200])
w4 = init_weights([200 * 5 * 85, 625])
w_o = init_weights([625, NUM_CLASSES]) 

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9).minimize(cost)
predict_softmax = tf.nn.softmax(py_x)
predict_op = tf.argmax(predict_softmax, 1)

training_errors = []
validation_errors = []

# get single examples
label, sample = decode_single("./data/singles.training.tfrecords")
v_label, v_sample = decode_single("./data/singles.validation.tfrecords")

# groups examples into batches randomly
samples_batch, labels_batch = tf.train.shuffle_batch([sample, label], batch_size=BATCH_SIZE, capacity=BATCH_SIZE * 2, min_after_dequeue=BATCH_SIZE)
test_samples_batch, test_labels_batch = tf.train.shuffle_batch([v_sample, v_label], batch_size=TEST_SIZE, capacity=TEST_SIZE, min_after_dequeue=0)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    tf.train.start_queue_runners(sess=sess)

    test_samples = test_samples_batch.eval(session=sess)
    test_labels = test_labels_batch.eval(session=sess)
    teX, teY = test_samples, one_hot(test_labels, 12)
    teX = teX.reshape(-1, IMG_X, IMG_Y, 1)
    print "Training batch size: %d samples" % BATCH_SIZE
    print "Test batch size: %d samples" % test_samples.shape[0]
    print "Learning rate: %f" % LEARNING_RATE
    print

    i = 0
    try:
        while True:
            i = i + 1
            print "Training round: %d (%s)" % (i, str(datetime.datetime.now().time()))

            # Load next batch of training samples
            trX = samples_batch.eval(session=sess)
            trY = one_hot(labels_batch.eval(session=sess), 12)
            trX = trX.reshape(-1, IMG_X, IMG_Y, 1)
            
            # Train using training batch
            sess.run(train_op, feed_dict={X: trX, Y: trY, p_keep_conv: 0.9, p_keep_hidden: 0.9})

            # Evaluate model
            print "  testing... (%s)" % (str(datetime.datetime.now().time())) 
            predict_teY = sess.run(predict_softmax, feed_dict={X: teX, Y: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0})
            predict_trY = sess.run(predict_softmax, feed_dict={X: trX, Y: trY, p_keep_conv: 1.0, p_keep_hidden: 1.0})
            prediction = np.argmax(predict_teY)  # sess.run(predict_op, feed_dict={X: teX, Y: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0})

            # Show results
            print "  evaluating... (%s)" % (str(datetime.datetime.now().time()))
            accuracy = np.mean(np.argmax(teY, axis=1) == prediction)
            test_loss = metrics.log_loss(teY, predict_teY)
            training_loss = metrics.log_loss(trY, predict_trY)
            print "  accuracy: %f, training loss: %f, test loss: %f" % (accuracy, training_loss, test_loss)
    except tf.errors.OutOfRangeError, e:
        print "All Done."

    sess.close()

"""
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()

    # Compute confusion matrix
    plt.subplot(1, 2, 2)
    predicted_labels = classifier.predict(test_batch_xs)
    target_labels = test_batch_ys
    cm = confusion_matrix(target_labels, predicted_labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("results.png")
"""