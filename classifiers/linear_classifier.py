from __future__ import print_function

import numpy as np
from classifiers.svm_loss import *
from classifiers.softmax_loss import *


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, Y, learning_rate=1e-3, reg=1e-5,
              num_iters=100, batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        """

        num_train, dim = X.shape
        num_classes = np.max(Y) + 1  # assume labels are 0...C-1

        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []

        for it in range(num_iters):

            ###################################################################
            # Sample batch_size from training data
            ###################################################################
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]

            ###################################################################
            # Compute loss and gradient
            ###################################################################
            loss, grad = self.loss(X_batch, Y_batch, reg)
            loss_history.append(loss)

            ###################################################################
            # Update weights W
            ###################################################################
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print("Iteration %d/%d: Loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use trained weights to predict labels.
        """
        scores = X.dot(self.W)
        Y_pred = np.argmax(scores, axis=1)
        return Y_pred

    def loss(self, X_batch, Y_batch, reg):
        """
        Placeholder — overridden in subclasses (SVM / Softmax)
        """
        raise NotImplementedError


class LinearSVM(LinearClassifier):
    """ Linear SVM classifier """

    def loss(self, X_batch, Y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, Y_batch, reg)


class LinearSoftmax(LinearClassifier):
    """ Linear Softmax classifier """

    def loss(self, X_batch, Y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, Y_batch, reg)
