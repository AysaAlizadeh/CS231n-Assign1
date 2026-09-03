import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    """

    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):

        # Compute scores
        scores = X[i].dot(W)

        # Numeric stability
        scores -= np.max(scores)

        # Softmax probabilities
        probs = np.exp(scores) / np.sum(np.exp(scores))

        # Loss
        loss += -np.log(probs[y[i]])

        # Gradient
        for j in range(num_classes):
            p = probs[j]
            if j == y[i]:
                dW[:, j] += (p - 1) * X[i]
            else:
                dW[:, j] += p * X[i]

    # Average and regularization
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    """

    num_train = X.shape[0]

    # Scores
    scores = X.dot(W)

    # Numeric stability
    scores -= np.max(scores, axis=1, keepdims=True)

    # Softmax probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Loss
    correct_logprobs = -np.log(probs[np.arange(num_train), y])
    loss = np.sum(correct_logprobs) / num_train
    loss += reg * np.sum(W * W)

    # Gradient
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1

    dW = X.T.dot(dscores)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW
