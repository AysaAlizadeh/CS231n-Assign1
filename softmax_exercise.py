#FILE NAME: softmax_exercise.py
########################################################################################
# Softmax exercise
########################################################################################

import random
import numpy as np
import matplotlib.pyplot as plt
from utilities.load_cifar10 import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    cifar10_dir = '/Users/aysa/Documents/TabrizUniversity/Deep Learning/Assign_1/cifar-10-batches-py'

    X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

    # subsample
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    Y_val = Y_train[mask]

    mask = list(range(num_training))
    X_train = X_train[mask]
    Y_train = Y_train[mask]

    mask = list(range(num_test))
    X_test = X_test[mask]
    Y_test = Y_test[mask]

    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    Y_dev = Y_train[mask]

    # reshape
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # subtract mean
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    # add bias
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev


try:
    del X_train, Y_train
    del X_test, Y_test
    print('Clear previously loaded data.')
except:
    pass

X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev = get_CIFAR10_data()

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', Y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', Y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)
print('dev data shape: ', X_dev.shape)
print('dev labels shape: ', Y_dev.shape)

########################################################################################
# Softmax Classifier
########################################################################################

from classifiers.softmax_loss import softmax_loss_naive

W = np.random.randn(3073, 10) * 0.0001
loss, grad = softmax_loss_naive(W, X_dev, Y_dev, 0.0)

print()
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))

loss, grad = softmax_loss_naive(W, X_dev, Y_dev, 0.0)

from utilities.gradient_check import grad_check_sparse

f = lambda w: softmax_loss_naive(w, X_dev, Y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

loss, grad = softmax_loss_naive(W, X_dev, Y_dev, 5e1)
f = lambda w: softmax_loss_naive(w, X_dev, Y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

########################################################################################
# Vectorized Softmax
########################################################################################

import time
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, Y_dev, 0.000005)
toc = time.time()
print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

from classifiers.softmax_loss import softmax_loss_vectorized
tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, Y_dev, 0.000005)
toc = time.time()
print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
print('Gradient difference: %f' % grad_difference)

########################################################################################
# Hyperparameter Tuning
########################################################################################

from classifiers.linear_classifier import LinearSoftmax

results = {}
best_val = -1
best_softmax = None

learning_rates = [3e-7, 5e-8, 9e-8]
regularization_strengths = [3e5, 5e5]

for lr in learning_rates:
    for reg in regularization_strengths:

        softmax = LinearSoftmax()

        loss_hist = softmax.train(
            X_train, Y_train,
            learning_rate=lr,
            reg=reg,
            num_iters=1500,
            verbose=False
        )

        Y_train_pred = softmax.predict(X_train)
        train_accuracy = np.mean(Y_train == Y_train_pred)

        Y_val_pred = softmax.predict(X_val)
        val_accuracy = np.mean(Y_val == Y_val_pred)

        results[(lr, reg)] = (train_accuracy, val_accuracy)

        if val_accuracy > best_val:
            best_val = val_accuracy
            best_softmax = softmax

# Print results
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e    reg %e    train accuracy: %f    val accuracy: %f' %
          (lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

########################################################################################
# Evaluate on Test Set
########################################################################################

Y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(Y_test == Y_test_pred)
print('softmax on raw pixels final test set accuracy: %f' % test_accuracy)

########################################################################################
# Visualize the learned weights
########################################################################################

w = best_softmax.W[:-1, :]
w = w.reshape(32, 32, 3, 10)

w_min, w_max = np.min(w), np.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(10):
    plt.subplot(2, 5, i + 1)
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])

plt.show()
