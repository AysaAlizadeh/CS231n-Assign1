import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, Y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1)        # ReLU
        scores = a1.dot(W2) + b2      # shape (N, C)

        if Y is None:
            return scores

        scores_shift = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shift)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_logprobs = -np.log(probs[np.arange(N), Y])
        data_loss = np.mean(correct_logprobs)
        reg_loss = reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        loss = data_loss + reg_loss

        grads = {}

        dscores = probs
        dscores[np.arange(N), Y] -= 1
        dscores /= N

        grads['W2'] = a1.T.dot(dscores) + 2 * reg * W2
        grads['b2'] = np.sum(dscores, axis=0)

        da1 = dscores.dot(W2.T)
        dz1 = da1.copy()
        dz1[z1 <= 0] = 0

        grads['W1'] = X.T.dot(dz1) + 2 * reg * W1
        grads['b1'] = np.sum(dz1, axis=0)

        return loss, grads

    def train(self, X, Y, X_val, Y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):

            batch_idx = np.random.choice(num_train, batch_size)
            X_batch = X[batch_idx]
            Y_batch = Y[batch_idx]

            loss, grads = self.loss(X_batch, Y_batch, reg)
            loss_history.append(loss)

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print(f"iteration {it}/{num_iters}: loss {loss}")

            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == Y_batch).mean()
                val_acc = (self.predict(X_val) == Y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1)
        scores = a1.dot(W2) + b2

        Y_pred = np.argmax(scores, axis=1)
        return Y_pred
