#FILE NAME: knn_exercise.py    #########################################################################################
# k-Nearest Neighbor (kNN) exercise
# Complete and hand in this code completed (including its outputs and any supporting code outside of it)

import numpy as np
import matplotlib.pyplot as plt
from utilities.load_cifar10 import load_CIFAR10

#######################################################################################################################

# Load the raw CIFAR-10 data.
# Change the following path to the directory that your "cifar-10-batches-py" file exists

cifar10_dir = '/Users/aysa/Documents/TabrizUniversity/Deep Learning/Assign_1/cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, Y_train
   del X_test, Y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)

#######################################################################################################################

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(Y_train == y)  #find train data indexes with class y
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

#######################################################################################################################

# Subsample the data for more efficient code execution in this exercise
num_training = 500   #5000
mask = list(range(num_training))
X_train = X_train[mask]
Y_train = Y_train[mask]

num_test = 50    #500
mask = list(range(num_test))
X_test = X_test[mask]
Y_test = Y_test[mask]

#######################################################################################################################

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

#######################################################################################################################

from classifiers.knn_classifier import KNearestNeighbor

# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
knn_model = KNearestNeighbor()
knn_model.train(X_train, Y_train)

#######################################################################################################################

# open classifiers/knn_classifier.py and implement the function "compute_distances_two_loops" that
# uses a (very inefficient) double loop over all pairs of (test, train) examples and
# computes the distance matrix one element at a time.

# Test your implementation:
dists = knn_model.compute_distances_two_loops(X_test)
print(dists.shape)

#######################################################################################################################

# We can visualize the distance matrix: each row is for a single test example and
# its distances to all training examples
plt.imshow(dists, interpolation='none')
plt.show()

#######################################################################################################################

# Now implement the function "predict_labels" in classifiers/knn_classifier.py
# and run the code below: We use k = 1 (which is Nearest Neighbor).
Y_test_pred = knn_model.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(Y_test_pred == Y_test)
accuracy = float(num_correct) / num_test
print('For k=1 we got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# Now lets try out a larger k, say k = 5:
Y_test_pred = knn_model.predict_labels(dists, k=5)
num_correct = np.sum(Y_test_pred == Y_test)
accuracy = float(num_correct) / num_test
print('For k=5 we got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

dists_one = knn_model.compute_distances_one_loop(X_test)

difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')


dists_two = knn_model.compute_distances_no_loops(X_test)

difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')


def time_function(f, *args):
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

two_loop_time = time_function(knn_model.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(knn_model.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(knn_model.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)


num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
Y_train_folds = []

X_train_folds = np.array_split(X_train, num_folds)
Y_train_folds = np.array_split(Y_train, num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = []
    for j in range(num_folds):
        X_val = X_train_folds[j]
        y_val = Y_train_folds[j]

        X_train_cv = np.vstack([fold for i, fold in enumerate(X_train_folds) if i != j])
        y_train_cv = np.hstack([fold for i, fold in enumerate(Y_train_folds) if i != j])

        knn_model = KNearestNeighbor()
        knn_model.train(X_train_cv, y_train_cv)

        y_pred = knn_model.predict(X_val, k=k)
        accuracy = np.mean(y_pred == y_val)

        k_to_accuracies[k].append(accuracy)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))


for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()


best_k = 3    # this is for num_train=500 and num_test=50

knn_model = KNearestNeighbor()
knn_model.train(X_train, Y_train)
Y_test_pred = knn_model.predict(X_test, k=best_k)

num_correct = np.sum(Y_test_pred == Y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


