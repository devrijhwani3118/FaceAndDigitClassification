import numpy as np
import random
import time
import copy

PIXEL_VALUES = {' ': 0, '+': 1, '#': 1}

def load_data(image_file, label_file, height, width):
    with open(label_file, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    with open(image_file, 'r') as f:
        lines = [line.rstrip('\n') for line in f]

    num_images = len(labels)
    X = []

    for i in range(num_images):
        start = i * height
        end = start + height
        image_lines = lines[start:end]
        image = []
        for line in image_lines:
            row = [PIXEL_VALUES.get(c, 0) for c in line.ljust(width)]
            image.extend(row)
        X.append(image)

    return np.array(X), np.array(labels)

class NeuralNet:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(hidden1_size, input_size) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((hidden1_size, 1))
        self.W2 = np.random.randn(hidden2_size, hidden1_size) * np.sqrt(1 / hidden1_size)
        self.b2 = np.zeros((hidden2_size, 1))
        self.W3 = np.random.randn(output_size, hidden2_size) * np.sqrt(1 / hidden2_size)
        self.b3 = np.zeros((output_size, 1))

    def relu(self, x): return np.maximum(0, x)
    def relu_derivative(self, x): return (x > 0).astype(float)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        a1 = self.relu(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = self.relu(z2)
        z3 = self.W3 @ a2 + self.b3
        a3 = self.softmax(z3)
        return a1, a2, a3

    def backward(self, x, y_true, a1, a2, a3):
        y = np.zeros_like(a3)
        y[y_true] = 1
        dz3 = a3 - y
        dW3 = dz3 @ a2.T
        db3 = dz3
        dz2 = (self.W3.T @ dz3) * self.relu_derivative(a2)
        dW2 = dz2 @ a1.T
        db2 = dz2
        dz1 = (self.W2.T @ dz2) * self.relu_derivative(a1)
        dW1 = dz1 @ x.T
        db1 = dz1

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        predictions = []
        for x in X:
            _, _, output = self.forward(x.reshape(-1, 1))
            predictions.append(np.argmax(output))
        return predictions

    def train(self, X, y, X_val, y_val, epochs=5):
        best_model = copy.deepcopy(self)
        best_val_acc = 0
        for epoch in range(epochs):
            for i in range(len(X)):
                x = X[i].reshape(-1, 1)
                a1, a2, a3 = self.forward(x)
                self.backward(x, y[i], a1, a2, a3)
            val_acc = evaluate(self, X_val, y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(self)
        return best_model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return np.mean(np.array(preds) == y_test)

def run_experiment(train_img, train_lbl, val_img, val_lbl, test_img, test_lbl,
                   num_classes, height, width, label="dataset"):

    X_train_full, y_train_full = load_data(train_img, train_lbl, height, width)
    X_val, y_val = load_data(val_img, val_lbl, height, width)
    X_test, y_test = load_data(test_img, test_lbl, height, width)
    input_size = height * width

    print(f"\n=== {label} ===")
    for pct in range(10, 101, 10):
        accs = []
        start_time = time.time()

        for _ in range(5):  # 5 runs
            indices = list(range(len(X_train_full)))
            random.shuffle(indices)
            subset_size = int(pct / 100 * len(indices))
            selected = indices[:subset_size]

            X_train = X_train_full[selected]
            y_train = y_train_full[selected]

            model = NeuralNet(input_size, 128, 64, num_classes, lr=0.01)
            best_model = model.train(X_train, y_train, X_val, y_val, epochs=5)
            acc = evaluate(best_model, X_test, y_test)
            accs.append(acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        elapsed = time.time() - start_time
        print(f"{pct}% data => Test Acc: {mean_acc:.4f} ± {std_acc:.4f}, Time: {elapsed:.2f}s")

# ---- Main ----
if __name__ == "__main__":
    run_experiment(
        "data/digitdata/trainingimages", "data/digitdata/traininglabels",
        "data/digitdata/validationimages", "data/digitdata/validationlabels",
        "data/digitdata/testimages", "data/digitdata/testlabels",
        num_classes=10, height=28, width=28, label="Digit Classification"
    )

    run_experiment(
        "data/facedata/facedatatrain", "data/facedata/facedatatrainlabels",
        "data/facedata/facedatavalidation", "data/facedata/facedatavalidationlabels",
        "data/facedata/facedatatest", "data/facedata/facedatatestlabels",
        num_classes=2, height=70, width=60, label="Face Classification"
    )