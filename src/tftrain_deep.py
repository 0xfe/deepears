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
TEST_SIZE=32

# Train
BATCH_SIZE=32
LEARNING_RATE=0.01
STEPS=100

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

def input_data(filename, batch_size):
    label, sample = decode_single(filename)
    samples_batch, labels_batch = tf.train.shuffle_batch([sample, label], batch_size=batch_size, capacity=batch_size * 2, min_after_dequeue=batch_size)
    return samples_batch, labels_batch

with tf.Session() as sess:
    classifier = tf.contrib.learn.DNNClassifier(
        n_classes=12,
        hidden_units=[1000,100],
        optimizer=tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE),
        dropout=0.1
    )


    test_samples_batch, test_labels_batch = input_data("./data/singles.validation.tfrecords", TEST_SIZE)
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    
    test_samples = test_samples_batch.eval(session=sess)
    test_labels = test_labels_batch.eval(session=sess)
    teX, teY = test_samples, one_hot(test_labels, 12)

    try:
        period = 0
        while True:
            print "Period: %d (%s)" % (period, str(datetime.datetime.now().time())) 
            # print sess.run(labels_batch) # this works
            # print sess.run(samples_batch) # this works
            classifier.partial_fit(input_fn=lambda: input_data("./data/singles.training.tfrecords", BATCH_SIZE), steps=STEPS)
            # predictions_training = classifier.predict_proba(training_features)
            predictions_validation = classifier.predict_proba(teX)
            #  log_loss_training = metrics.log_loss(training_labels, predictions_training)
            log_loss_validation = metrics.log_loss(teY, predictions_validation)
            accuracy_validation = 0 # metrics.accuracy_score(test_labels, predictions_validation)
            #  training_errors.append(log_loss_training)
            print "  loss=%f, accuracy=%f" % (log_loss_validation, accuracy_validation)
            period = period + 1
    except tf.errors.OutOfRangeError, e:
            print "All Done."

    sess.close()

"""
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