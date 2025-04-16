import numpy as np
import random
import os
import time
import copy

# Constants
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 70
PIXEL_VALUES = {' ': 0, '+': 1, '#': 1}

# ---- 1. Load Data ----
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




# ---- 2. Neural Network ----
class NeuralNet:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(hidden1_size, input_size) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((hidden1_size, 1))

        self.W2 = np.random.randn(hidden2_size, hidden1_size) * np.sqrt(1 / hidden1_size)
        self.b2 = np.zeros((hidden2_size, 1))

        self.W3 = np.random.randn(output_size, hidden2_size) * np.sqrt(1 / hidden2_size)
        self.b3 = np.zeros((output_size, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

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
            
            # Evaluate after each epoch
            val_acc = evaluate(self, X_val, y_val)
            print(f"  Epoch {epoch+1}/{epochs} - Val Accuracy: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(self)

        return best_model

# ---- 3. Evaluation ----
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    accuracy = np.mean(np.array(preds) == y_test)
    return accuracy

# ---- 4. Main Experiment Runner ----
def run_experiment(train_img, train_lbl, val_img, val_lbl, test_img, test_lbl, num_classes, height, width, label="dataset"):
    X_train, y_train = load_data(train_img, train_lbl, height, width)
    X_val, y_val = load_data(val_img, val_lbl, height, width)
    X_test, y_test = load_data(test_img, test_lbl, height, width)
    input_size = height * width

    print(f"\n=== Training on {label} ===")
    model = NeuralNet(input_size, 128, 64, num_classes, lr=0.01)
    best_model = model.train(X_train, y_train, X_val, y_val, epochs=5)

    test_acc = evaluate(best_model, X_test, y_test)
    print(f">>> Final Test Accuracy on {label}: {test_acc:.4f}")


# ---- 5. Entry Point ----
if __name__ == "__main__":
    # Digits: 28x28
    run_experiment(
        "data/digitdata/trainingimages", "data/digitdata/traininglabels",
        "data/digitdata/validationimages", "data/digitdata/validationlabels",
        "data/digitdata/testimages", "data/digitdata/testlabels",
        num_classes=10, height=28, width=28, label="Digit Classification"
    )

    # Faces: 70x60
    run_experiment(
        "data/facedata/facedatatrain", "data/facedata/facedatatrainlabels",
        "data/facedata/facedatavalidation", "data/facedata/facedatavalidationlabels",
        "data/facedata/facedatatest", "data/facedata/facedatatestlabels",
        num_classes=2, height=70, width=60, label="Face Classification"
    )



# import numpy as np
# import random
# import os
# import time

# # Constants
# IMAGE_WIDTH = 60
# IMAGE_HEIGHT = 70
# PIXEL_VALUES = {' ': 0, '+': 1, '#': 1}  # Binary features

# # ---- 1. Load Data ----
# def load_data(image_file, label_file):
#     # Read and strip the image file content to remove extra whitespace
#     with open(image_file, 'r') as f:
#         raw_data = f.read().strip()  # Remove any leading/trailing whitespace
#         images = raw_data.split('\n\n')
    
#     # Read labels, also filtering out any empty lines
#     with open(label_file, 'r') as f:
#         labels = [int(line.strip()) for line in f if line.strip()]

#     X = []
#     for img in images:
#         # Remove any leading/trailing whitespace from each image block
#         img = img.strip()
#         if not img:  # Skip empty entries
#             continue
#         # Remove newline characters inside an image block and convert to pixel values
#         pixels = [PIXEL_VALUES.get(c, 0) for c in img.replace('\n', '')]
#         # Only add the image if it has the expected number of pixels
#         if len(pixels) == IMAGE_WIDTH * IMAGE_HEIGHT:
#             X.append(pixels)
    
#     # Convert to a NumPy array and ensure it's 2D (each row is an image)
#     X = np.array(X)
#     if X.ndim == 1:
#         X = X.reshape(-1, IMAGE_WIDTH * IMAGE_HEIGHT)
    
#     return X, np.array(labels)


# # ---- 2. Define Neural Network ----
# class NeuralNet:
#     def __init__(self, input_size, hidden1_size, hidden2_size, output_size, lr=0.01):
#         self.lr = lr
#         # Xavier Initialization
#         self.W1 = np.random.randn(hidden1_size, input_size) * np.sqrt(1 / input_size)
#         self.b1 = np.zeros((hidden1_size, 1))

#         self.W2 = np.random.randn(hidden2_size, hidden1_size) * np.sqrt(1 / hidden1_size)
#         self.b2 = np.zeros((hidden2_size, 1))

#         self.W3 = np.random.randn(output_size, hidden2_size) * np.sqrt(1 / hidden2_size)
#         self.b3 = np.zeros((output_size, 1))

#     def relu(self, x):
#         return np.maximum(0, x)

#     def relu_derivative(self, x):
#         return (x > 0).astype(float)

#     def softmax(self, x):
#         e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
#         return e_x / np.sum(e_x, axis=0, keepdims=True)

#     def forward(self, x):
#         z1 = self.W1 @ x + self.b1
#         a1 = self.relu(z1)
#         z2 = self.W2 @ a1 + self.b2
#         a2 = self.relu(z2)
#         z3 = self.W3 @ a2 + self.b3
#         a3 = self.softmax(z3)

#         return a1, a2, a3

#     def backward(self, x, y_true, a1, a2, a3):
#         # One-hot encode y
#         y = np.zeros_like(a3)
#         y[y_true] = 1

#         dz3 = a3 - y
#         dW3 = dz3 @ a2.T
#         db3 = dz3

#         dz2 = (self.W3.T @ dz3) * self.relu_derivative(a2)
#         dW2 = dz2 @ a1.T
#         db2 = dz2

#         dz1 = (self.W2.T @ dz2) * self.relu_derivative(a1)
#         dW1 = dz1 @ x.T
#         db1 = dz1

#         # Gradient descent update
#         self.W3 -= self.lr * dW3
#         self.b3 -= self.lr * db3
#         self.W2 -= self.lr * dW2
#         self.b2 -= self.lr * db2
#         self.W1 -= self.lr * dW1
#         self.b1 -= self.lr * db1

#     def predict(self, X):
#         predictions = []
#         for x in X:
#             _, _, output = self.forward(x.reshape(-1, 1))
#             predictions.append(np.argmax(output))
#         return predictions

#     def train(self, X, y, epochs=5):
#         for _ in range(epochs):
#             for i in range(len(X)):
#                 x = X[i].reshape(-1, 1)
#                 a1, a2, a3 = self.forward(x)
#                 self.backward(x, y[i], a1, a2, a3)

# # ---- 3. Evaluation ----
# def evaluate(model, X_test, y_test):
#     preds = model.predict(X_test)
#     accuracy = np.mean(np.array(preds) == y_test)
#     return accuracy

# # ---- 4. Main Runner ----
# def run_experiment(image_path, label_path, num_classes, percentages=[10,20,30,40,50,60,70,80,90,100]):
#     X, y = load_data(image_path, label_path)
#     input_size = X.shape[1]

#     for percent in percentages:
#         accuracies = []
#         start = time.time()

#         for _ in range(5):  # repeat 5 times
#             indices = list(range(len(X)))
#             random.shuffle(indices)
#             split = int(percent / 100 * len(X))
#             train_idx, test_idx = indices[:split], indices[split:]

#             X_train, y_train = X[train_idx], y[train_idx]
#             X_test, y_test = X[test_idx], y[test_idx]

#             model = NeuralNet(input_size, 128, 64, num_classes, lr=0.01)
#             model.train(X_train, y_train, epochs=3)
#             acc = evaluate(model, X_test, y_test)
#             accuracies.append(acc)

#         duration = time.time() - start
#         print(f"{percent}% data => Acc: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}, Time: {duration:.2f}s")

# # ---- 5. Entry Point ----
# if __name__ == "__main__":
#     print("Running on digitdata...")
#     run_experiment("data/digitdata/trainingimages", "data/digitdata/traininglabels", num_classes=10)

#     print("Running on facedata...")
#     run_experiment("data/facedata/facedatatrain", "data/facedata/facedatatrainlabels", num_classes=2)


