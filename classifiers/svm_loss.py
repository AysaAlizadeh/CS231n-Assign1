# FILE NAME: svm_loss.py

import numpy as np
from random import shuffle


def svm_loss_naive(W, X, Y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).
  """

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros(W.shape)

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[Y[i]]

        for j in range(num_classes):
            if j == Y[i]:
                continue
            margin = scores[j] - correct_class_score + 1

            if margin > 0:
                loss += margin

                # gradient update
                dW[:, j] += X[i]
                dW[:, Y[i]] -= X[i]

    # Average over all training examples
    loss /= num_train
    dW /= num_train

    # Add regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, Y, reg):
    """
  Structured SVM loss function, vectorized implementation.
  """

    num_train = X.shape[0]

    # Compute scores
    scores = X.dot(W)

    # Correct class scores
    correct_scores = scores[np.arange(num_train), Y].reshape(-1, 1)

    # Compute margins
    margins = np.maximum(0, scores - correct_scores + 1)
    margins[np.arange(num_train), Y] = 0

    # Loss
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    # Gradient
    binary = (margins > 0).astype(float)
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), Y] = -row_sum

    dW = X.T.dot(binary)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW
