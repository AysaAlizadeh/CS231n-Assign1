from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import random
from classifiers.neural_net_classifier import TwoLayerNet


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


input_size = 4      # D: input data dimension
hidden_size = 10    # H
num_classes = 3     # C
num_inputs = 5      # N

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)   # N * D
    Y = np.array([0, 1, 2, 2, 1])
    return X, Y

net = init_toy_model()
X, Y = init_toy_data()


scores = net.loss(X)         # without Y parameter loss function only returns scores
print('Your scores:')
print(scores)
print()
print('correct scores:')

correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# The difference should be very small. We get < 1e-7
print('Sum of differences between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))
print()

loss, _ = net.loss(X, Y, reg=0.05)
correct_loss = 1.30378789133

# The difference should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.abs(loss - correct_loss))
print()

from utilities.gradient_check import eval_numerical_gradient

loss, grads = net.loss(X, Y, reg=0.05)

# these should all be less than 1e-8 or so.
for param_name in grads:
    f = lambda W: net.loss(X, Y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))
print()


net = init_toy_model()
stats = net.train(X, Y, X, Y,       # in toy example X_val=X-train
                  learning_rate=1e-1, reg=5e-6,
                  num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])
print()

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()

from utilities.load_cifar10 import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '/Users/aysa/Documents/TabrizUniversity/Deep Learning/Assign_1/cifar-10-batches-py'

    X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    Y_val = Y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    Y_train = Y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    Y_test = Y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

try:
    del X_train, Y_train
    del X_test, Y_test
    print('Clear previously loaded data.')
except:
    pass

X_train, Y_train, X_val, Y_val, X_test, Y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', Y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', Y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)
print()

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, Y_train, X_val, Y_val,
                  num_iters=1000, batch_size=200,
                  learning_rate=1e-4, learning_rate_decay=0.95,
                  reg=0.25, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == Y_val).mean()
print('Final validation accuracy: ', val_acc)
print()


def plot_loss_acc(stats):
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.show()

plot_loss_acc(stats)

from utilities.vis_utils import visualize_grid


def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)   # ?? -1 becomes H=50
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(net)

best_net = None
best_val = 0

learning_rates = [1e-3, 5e-4, 1e-4]
hidden_sizes = [50, 100, 150]
regs = [0.25, 0.1, 0.01]

print("\n--- Hyperparameter search started ---\n")

for lr in learning_rates:
    for hs in hidden_sizes:
        for reg in regs:
            print(f"Training with lr={lr}, hidden={hs}, reg={reg}")

            net = TwoLayerNet(input_size, hs, num_classes)
            stats = net.train(
                X_train, Y_train, X_val, Y_val,
                num_iters=1500,
                batch_size=200,
                learning_rate=lr,
                learning_rate_decay=0.95,
                reg=reg,
                verbose=False
            )

            val_acc = (net.predict(X_val) == Y_val).mean()
            print("  val_acc = ", val_acc)

            if val_acc > best_val:
                best_val = val_acc
                best_net = net

print("\nBest validation accuracy:", best_val)

show_net_weights(best_net)


test_acc = (best_net.predict(X_test) == Y_test).mean()
print('Test accuracy: ', test_acc)

