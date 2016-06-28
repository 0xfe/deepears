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
LEARNING_RATE=0.3
STEPS=10000

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver

def now():
    return str(datetime.datetime.now().time())

class EarMonitor(tf.contrib.learn.monitors.EveryN):
  """Runs evaluation of the Estimator every n steps.
  Can do early stopping on validation metrics if
  `early_stopping_rounds` provided.
  """

  def __init__(self, x=None, y=None, input_fn=None, batch_size=None,
               eval_steps=None,
               every_n_steps=100, metrics=None, early_stopping_rounds=None,
               early_stopping_metric="loss",
               early_stopping_metric_minimize=True, name=None):
    """Initializes ValidationMonitor.
    Args:
      x: matrix or tensor of shape [n_samples, n_features...]. Can be
         iterator that returns arrays of features. The training input
         samples for fitting the model. If set, `input_fn` must be `None`.
      y: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
         iterator that returns array of targets. The training target values
         (class labels in classification, real numbers in regression). If set,
         `input_fn` must be `None`.
      input_fn: Input function. If set, `x`, `y`, and `batch_size` must be
          `None`.
      batch_size: minibatch size to use on the input, defaults to first
          dimension of `x`. Must be `None` if `input_fn` is provided.
      eval_steps: Number of steps to run evaluation. `None` means to run
          until records finish.
      every_n_steps: Runs this monitor every N steps.
      metrics: Dict of metric ops to run. If None, the default metric functions
        are used; if {}, no metrics are used.
      early_stopping_rounds: If validation metric didn't go down for this many
          steps, then stop training.
      early_stopping_metric: `str`, name of the metric to early stop.
      early_stopping_metric_minimize: `bool`, True if minimize, False
          if maximize. For example, minimize `loss` or `mean_squared_error` and
          maximize `accuracy` or `f1`.
      name: `str`, appended to output sub-folder. If None uses `eval`
          sub-folder, else, `eval-%name%` is used to save sum.
    Raises:
      ValueError: If both x and input_fn are provided.
    """
    super(EarMonitor, self).__init__(every_n_steps=every_n_steps,
                                            first_n_steps=-1)
    if x is None and input_fn is None:
      raise ValueError("Either x or input_fn should be provided.")
    self.x = x
    self.y = y
    self.input_fn = input_fn
    self.batch_size = batch_size
    self.eval_steps = eval_steps
    self.metrics = metrics
    self.early_stopping_rounds = early_stopping_rounds
    self.early_stopping_metric = early_stopping_metric
    self.early_stopping_metric_minimize = early_stopping_metric_minimize
    self.name = name
    self._best_value_step = None
    self._best_value = None
    self._early_stopped = False
    self._latest_path = None
    self._latest_path_step = None

  @property
  def early_stopped(self):
    return self._early_stopped

  @property
  def best_step(self):
    return self._best_value_step

  @property
  def best_value(self):
    return self._best_value

  def every_n_step_end(self, step, outputs):
    super(EarMonitor, self).every_n_step_end(step, outputs)
    if self._estimator is None:
      raise ValueError("Missing call to set_estimator.")
    # Check that we are not running evaluation on the same checkpoint.
    latest_path = saver.latest_checkpoint(self._estimator.model_dir)
    if latest_path == self._latest_path:
      logging.info("Skipping evaluation due to same checkpoint %s for step %d "
                   "as for step %d.", latest_path, step, self._latest_path_step)
      return False
    self._latest_path = latest_path
    self._latest_path_step = step

    # Run evaluation and log it.
    outputs = self._estimator.evaluate(
        x=self.x, y=self.y, input_fn=self.input_fn, batch_size=self.batch_size,
        steps=self.eval_steps, metrics=self.metrics, name=self.name)
    stats = []
    for name in outputs:
      stats.append("%s = %s" % (name, str(outputs[name])))
    line = "%s (%s): evaluation (step %d): %s" % (now(), self.name, step, ", ".join(stats))
    logging.info(line)
    print line

    # Early stopping logic.
    if self.early_stopping_rounds is not None:
      if (self._best_value is None or
          (self.early_stopping_metric_minimize and
           outputs[self.early_stopping_metric] < self._best_value) or
          (not self.early_stopping_metric_minimize and
           outputs[self.early_stopping_metric] > self._best_value)):
        self._best_value = outputs[self.early_stopping_metric]
        self._best_value_step = step
      stop_now = (step - self._best_value_step >= self.early_stopping_rounds)
      if stop_now:
        logging.info("Stopping. Best step: {} with {} = {}."
                     .format(self._best_value_step,
                             self.early_stopping_metric, self._best_value))
        self._early_stopped = True
        return True
    return False


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
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    try:
        print "Training (%s):" % now()
        classifier = tf.contrib.learn.DNNClassifier(
            n_classes=12,
            hidden_units=[1000, 300],
            optimizer=tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE),
            dropout=0.1,
            config=tf.contrib.learn.RunConfig(num_cores=16)
        )
        classifier.fit(input_fn=training_data, steps=STEPS,
            monitors=[EarMonitor(input_fn=test_data, name="validation", eval_steps=100),
                      EarMonitor(input_fn=training_data, name="training", eval_steps=100)])
    except tf.errors.OutOfRangeError, e:
            print "All Done."

    results = classifier.evaluate(input_fn=test_data, steps=1000)
    print "    Test:", pformat(results)
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