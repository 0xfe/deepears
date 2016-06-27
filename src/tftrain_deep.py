#!/usr/bin/env python
#
# $ source tf/bin/activate
# $ ./tftrain.py

import tensorflow as tf
import numpy as np
import scipy, numpy as np
from scipy.io import wavfile
import glob
from os import path
import re
import random
import sys

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

# Validation
TEST_SIZE=1000

# Train
BATCH_SIZE=100
LEARNING_RATE=10

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

# get single examples
label, sample = decode_single("./data/singles.training.tfrecords")
v_label, v_sample = decode_single("./data/singles.validation.tfrecords")

# groups examples into batches randomly
samples_batch, labels_batch = tf.train.shuffle_batch([sample, label], batch_size=BATCH_SIZE, capacity=BATCH_SIZE * 2, min_after_dequeue=BATCH_SIZE)
test_samples_batch, test_labels_batch = tf.train.shuffle_batch([v_sample, v_label], batch_size=TEST_SIZE, capacity=TEST_SIZE, min_after_dequeue=0)

def input_data():
    return samples_batch, labels_batch

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden):
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)
    return tf.matmul(h2, w_o)

X = tf.placeholder("float", [None, NUM_INPUTS])
Y = tf.placeholder("float", [None, 12])

# NN layers (inputs, outputs)
w_h = init_weights([NUM_INPUTS, 10000])
w_h2 = init_weights([10000, 625])
w_o = init_weights([625, 12])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(cost)
predict_softmax = tf.nn.softmax(py_x)
predict_op = tf.argmax(py_x, 1)

training_errors = []
validation_errors = []

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    tf.train.start_queue_runners(sess=sess)

    test_samples = test_samples_batch.eval(session=sess)
    test_labels = test_labels_batch.eval(session=sess)
    teX, teY = test_samples, one_hot(test_labels, 12)

    print "Test shape (samples, labels):", test_samples.shape, test_labels.shape

    i = 0
    try:
        while True:
            i = i + 1

            # Load next batch of training samples
            trX = samples_batch.eval(session=sess)
            trY = one_hot(labels_batch.eval(session=sess), 12)
            
            # Train using training batch, and predict test batch
            sess.run(train_op, feed_dict={X: trX, Y: trY, p_keep_input: 0.9, p_keep_hidden: 0.9})
            test_prediction = sess.run(predict_op, feed_dict={X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden: 1.0}) 

            # Evaluate model
            accuracy = np.mean(np.argmax(teY, axis=1) == test_prediction)
            test_loss = metrics.log_loss(teY, sess.run(predict_softmax, feed_dict={X: teX, Y: teY, p_keep_input: 1.0, p_keep_hidden: 1.0}))
            training_loss = metrics.log_loss(trY, sess.run(predict_softmax, feed_dict={X: trX, Y: trY, p_keep_input: 1.0, p_keep_hidden: 1.0}))
            print "Round: ", i
            print "  accuracy: %f, training loss: %f, test loss: %f" % (accuracy, training_loss, test_loss)
    except tf.errors.OutOfRangeError, e:
        print "All Done."

    sess.close()

"""
for period in range(0,10):
        print "Period: ", period
        print sess.run(labels_batch)
        print sess.run(samples_batch)
        classifier.fit(input_fn=input_data, steps=STEPS)


        # predictions_training = classifier.predict_proba(samples_batch)
#        predictions_validation = classifier.predict_proba(test_batch_xs)

        # log_loss_training = metrics.log_loss(one_hot(labels_batch, 12), predictions_training)
#        log_loss_validation = metrics.log_loss(one_hot(test_batch_ys, 12), predictions_validation)
#             training_errors.append(log_loss_training)
#        validation_errors.append(log_loss_validation)
        # print "  period %02d : loss %3.2f" % (period, log_loss_training) 
#        final_predictions = classifier.predict(test_batch_xs)
#        accuracy_validation = metrics.accuracy_score(test_batch_ys, final_predictions)
#        print "    test accuracy: %0.2f" % accuracy_validation

    sys.exit(0)

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