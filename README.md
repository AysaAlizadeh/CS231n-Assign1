# CS231n Assignment 1 — Image Classification on CIFAR-10

Implementation of four classic image classifiers from scratch using NumPy, trained and evaluated on the **CIFAR-10** dataset:

1. **K-Nearest Neighbors (KNN)**
2. **Multiclass SVM** (linear classifier with hinge loss)
3. **Softmax classifier** (linear classifier with cross-entropy loss)
4. **Two-Layer Neural Network**

## 📁 Project Structure

```
CS231n-Assign1/
├── knn_exercise.py                # Main script for KNN
├── svm_exercise.py                # Main script for SVM
├── softmax_exercise.py            # Main script for Softmax
├── two_layer_nn_exercise.py       # Main script for Two-Layer NN
├── classifiers/
│   ├── knn_classifier.py          # KNearestNeighbor class
│   ├── linear_classifier.py       # LinearClassifier, LinearSVM, LinearSoftmax
│   ├── svm_loss.py                # SVM hinge loss (naive + vectorized)
│   ├── softmax_loss.py            # Softmax cross-entropy loss (naive + vectorized)
│   └── neural_net_classifier.py   # TwoLayerNet class
├── utilities/
│   └── load_cifar10.py            # CIFAR-10 dataset loader
├── Figure_1.png / Figure_2.png / Figure_3.png   # Generated plots
├── .gitignore
└── README.md
```

## ⚙️ Setup

1. Download the CIFAR-10 dataset (Python version) from the [official site](https://www.cs.toronto.edu/~kriz/cifar.html) and unpack it to get the `cifar-10-batches-py` folder.
2. Update the dataset path in `knn_exercise.py`, `load_cifar10.py`, and the other exercise scripts to point to your local `cifar-10-batches-py` folder.
3. Install dependencies:

```bash
pip install numpy matplotlib
```

4. Run any exercise:

```bash
python knn_exercise.py
python svm_exercise.py
python softmax_exercise.py
python two_layer_nn_exercise.py
```

---

## 1️⃣ K-Nearest Neighbors

`classifiers/knn_classifier.py` implements the `KNearestNeighbor` class with three distance-computation strategies:

- **Two-loop** — nested loop over test and train samples (slowest, most explicit)
- **One-loop** — vectorized over training samples, looped over test samples
- **No-loop** — fully vectorized using matrix operations (fastest)

Predictions are made by majority vote among the `k` nearest neighbors (L2 distance).

### Results

**Dataset shapes:**

| Set | Shape |
|---|---|
| Training data | (50000, 32, 32, 3) |
| Test data | (10000, 32, 32, 3) |

A subset of 500 training / 50 test samples was used for the experiments.

**Distance computation timing:**

| Method | Time (s) |
|---|---|
| Two loops | 0.134830 |
| One loop | 0.100096 |
| No loops | 0.005218 |

All three implementations produced identical distance matrices ✅

**5-fold cross-validation accuracy per k:**

| k | Avg. accuracy |
|---|---|
| 1 | 0.250 |
| 3 | 0.246 |
| 5 | 0.224 |
| 8 | 0.210 |
| 10 | 0.230 |
| 12 | 0.214 |
| 15 | 0.220 |
| 20 | 0.216 |
| 50 | 0.214 |
| 100 | 0.198 |

**Best k found:** `k = 1`
**Final test accuracy (k=1):** `12 / 50 correct → 24.0%`

> Accuracy is low because this run used a small subset (500 train / 50 test) for speed, as is standard for this exercise.

---

## 2️⃣ Multiclass SVM

`classifiers/svm_loss.py` implements the structured (hinge) SVM loss with two versions:

- `svm_loss_naive` — explicit loops over training examples and classes
- `svm_loss_vectorized` — fully vectorized with NumPy broadcasting

The loss uses a margin of 1 between the correct class score and other class scores, plus L2 regularization on the weights. `classifiers/linear_classifier.py` wraps this in a `LinearSVM` class trained via mini-batch stochastic gradient descent (SGD).

---

## 3️⃣ Softmax Classifier

`classifiers/softmax_loss.py` implements the cross-entropy (softmax) loss with two versions:

- `softmax_loss_naive` — explicit loops with per-class gradient accumulation
- `softmax_loss_vectorized` — fully vectorized using matrix operations

Numeric stability is handled by subtracting the max score before exponentiating. `classifiers/linear_classifier.py` wraps this in a `LinearSoftmax` class, also trained via mini-batch SGD.

---

## 4️⃣ Two-Layer Neural Network

`classifiers/neural_net_classifier.py` implements a `TwoLayerNet`:

- **Architecture:** Input → Fully Connected → ReLU → Fully Connected → Softmax
- **Forward pass:** computes class scores and, when labels are provided, the softmax loss + L2 regularization
- **Backward pass:** manually derived gradients for `W1, b1, W2, b2` via backpropagation
- **Training:** mini-batch SGD with learning rate decay, tracking loss/train-accuracy/val-accuracy history per epoch

---

## 🛠️ Tech Stack

- Python
- NumPy
- Matplotlib (for cross-validation plots, sample image grids, loss curves)

## 👤 Author

Aysa Alizadeh
