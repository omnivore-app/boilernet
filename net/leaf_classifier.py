#! /usr/bin/python3


import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm


def weightedLoss(originalLossFunc, weightsList):
    def lossFunc(true, pred):

        # axis = -1  # if channels last
        # axis = 1  # if channels first

        # argmax returns the index of the element with the greatest value
        # done in the class axis, it returns the class index
        # classSelectors = tf.argmax(true, axis=axis)
        # if your loss is sparse, use only true as classSelectors

        # considering weights are ordered by class, for each class
        # true(1) if the class index is equal to the weight index
        # convert i to int64
        # classSelectors = [tf.equal(tf.cast(i, tf.int64), classSelectors)
        #                   for i in range(len(weightsList))]

        classSelectors = [False, True]

        # casting boolean to float for calculations
        # each tensor in the list contains 1 where ground true class is equal to its index
        # if you sum all these, you will get a tensor full of ones.
        classSelectors = [tf.cast(x, tf.float32)
                          for x in classSelectors]

        # for each of the selections above, multiply their respective weight
        weights = [sel * w for sel, w in zip(classSelectors, weightsList)]

        # sums all the selections
        # result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]

        # make sure your originalLossFunc only collapses the class axis
        # you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true, pred)
        loss = loss * weightMultiplier

        return loss
    return lossFunc


class Metrics(tf.keras.callbacks.Callback):
    """Calculate metrics for a dev-/testset and add them to the logs."""

    def __init__(self, clf, data, steps, interval, prefix=''):
        self.clf = clf
        self.data = data
        self.steps = steps
        self.interval = interval
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.interval == 0:
            y_true, y_pred = self.clf.eval(
                self.data, self.steps, desc=self.prefix)
            p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
        else:
            p, r, f, s = np.nan, np.nan, np.nan, np.nan
        logs_new = {'{}_precision'.format(self.prefix): p,
                    '{}_recall'.format(self.prefix): r,
                    '{}_f1'.format(self.prefix): f,
                    '{}_support'.format(self.prefix): s}
        logs.update(logs_new)


class Saver(tf.keras.callbacks.Callback):
    """Save the model."""

    def __init__(self, path, interval):
        self.path = path
        self.interval = interval

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.interval == 0:
            file_name = os.path.join(
                self.path, 'model.{:03d}.h5'.format(epoch))
            self.model.save(file_name)


# pylint: disable=E1101
class LeafClassifier(object):
    """This classifier assigns labels to sequences based on words and HTML tags."""

    def __init__(self, input_size, num_layers, hidden_size, dropout, dense_size, class_weights):
        """Construct the network."""
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dense_size = dense_size
        self.class_weights = class_weights
        self.model = self._get_model()

    def _get_model(self):
        """Return a keras model."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(
            input_shape=(None, self.input_size)))
        model.add(tf.keras.layers.Dense(self.dense_size, activation='relu'))
        model.add(tf.keras.layers.Masking(mask_value=0))
        for _ in range(self.num_layers):
            lstm = tf.keras.layers.LSTM(
                self.hidden_size, return_sequences=True)
            model.add(tf.keras.layers.Bidirectional(lstm))
        model.add(tf.keras.layers.Dropout(self.dropout))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(loss=weightedLoss(
            tf.keras.losses.binary_crossentropy, self.class_weights), optimizer='adam')
        return model

    def train(self, train_dataset, train_steps, epochs, log_file, ckpt,
              dev_dataset=None, dev_steps=None, test_dataset=None, test_steps=None, interval=1):
        """Train a number of input sequences."""
        callbacks = [Saver(ckpt, interval)]
        if dev_dataset is not None:
            callbacks.append(
                Metrics(self, dev_dataset, dev_steps, interval, 'dev'))
        if test_dataset is not None:
            callbacks.append(Metrics(self, test_dataset,
                             test_steps, interval, 'test'))
        callbacks.append(tf.keras.callbacks.CSVLogger(log_file))

        self.model.fit(train_dataset, steps_per_epoch=train_steps, epochs=epochs,
                       callbacks=callbacks)

    def eval(self, dataset, steps, desc=None):
        """Evaluate the model on the test data and return the metrics."""
        y_true, y_pred = [], []
        for b_x, b_y in tqdm(dataset, total=steps, desc=desc):
            # somehow this cast is necessary
            b_x = tf.dtypes.cast(b_x, 'float32')

            y_true.extend(b_y.numpy().flatten())
            y_pred.extend(
                np.around(self.model.predict_on_batch(b_x)).flatten())
        return y_true, y_pred
