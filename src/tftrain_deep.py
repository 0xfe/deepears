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
import datetime
from pprint import pprint as pp
from pprint import pformat

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
NUM_INPUTS = IMG_X * IMG_Y # Total number of inputs

# Validation
TEST_SIZE=32

# Train
BATCH_SIZE=32
LEARNING_RATE=0.1
STEPS=10000

# One-hot encode values in batch.
def one_hot(batch, max_value):
    b = np.zeros((len(batch), max_value), dtype='int32')
    b[np.arange(len(batch)), batch] = 1
    return b

def decode_single(filename):
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'sample': tf.FixedLenFeature([NUM_INPUTS], tf.float32)
        })
    # now return the converted data
    label = features['label']
    sample = features['sample']
    return label, sample

def input_data(filename, batch_size):
    label, sample = decode_single(filename)
    samples_batch, labels_batch = tf.train.shuffle_batch([sample, label], batch_size=batch_size, capacity=batch_size * 2, min_after_dequeue=batch_size)
    return samples_batch, labels_batch

def training_data():
    return input_data("./data/singles.training.tfrecords", BATCH_SIZE)  

def test_data():
    return input_data("./data/singles.validation.tfrecords", TEST_SIZE)

with tf.Session() as sess:
    test_samples_batch, test_labels_batch = test_data()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    test_samples = test_samples_batch.eval(session=sess)
    test_labels = test_labels_batch.eval(session=sess)
    teX, teY = test_samples, one_hot(test_labels, 12)

    try:
        classifier = tf.contrib.learn.DNNClassifier(
            n_classes=12,
            hidden_units=[1000, 100],
            optimizer=tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE),
            dropout=0.1,
            config=tf.contrib.learn.RunConfig(num_cores=16)
        )
        period = 0
        while True:
            print "Period: %d (%s)" % (period, str(datetime.datetime.now().time())) 
            classifier.partial_fit(input_fn=training_data, steps=STEPS)
            results = classifier.evaluate(input_fn=training_data, steps=100)
            print "    Training:", pformat(results)
            results = classifier.evaluate(input_fn=test_data, steps=100)
            print "    Test:", pformat(results)
            period = period + 1
    except tf.errors.OutOfRangeError, e:
            print "All Done."

    sess.close()

"""
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