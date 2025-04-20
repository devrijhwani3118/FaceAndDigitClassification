import numpy as np
import random
import time
import copy

PIXEL_VALUES = {' ': 0, '+': 1, '#': 1}

def load_data(image_file, label_file, height, width):
    labels=[]
    lines=[]
    flattened_image = []
    with open(label_file, 'r') as f:
        for line in f:
          if line.strip():
            clean_line = line.strip()
            labels.append(int(clean_line))

    with open(image_file, 'r') as f:
        for line in f:
            img_line=line.rstrip('\n')
            lines.append(img_line)

    num_images = len(labels)

    for i in range(num_images):
        start = i * height
        end = start + height
        image_lines = lines[start:end]
        image = []
        for line in image_lines:
            row = []
            for c in line.ljust(width):
              value = PIXEL_VALUES.get(c, 0)
              row.append(value)
            image.extend(row)
        flattened_image.append(image)

    return np.array(flattened_image), np.array(labels)

class NeuralNet:
    def __init__(self, input_size, hidden_layer_1_size, hidden_layer_2_size, output_layer_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.Weight1 = np.random.randn(hidden_layer_1_size, input_size) * np.sqrt(1 / input_size)
        self.bias1 = np.zeros((hidden_layer_1_size, 1))
        self.Weight2 = np.random.randn(hidden_layer_2_size, hidden_layer_1_size) * np.sqrt(1 / hidden_layer_1_size)
        self.bias2 = np.zeros((hidden_layer_2_size, 1))
        self.Weight3 = np.random.randn(output_layer_size, hidden_layer_2_size) * np.sqrt(1 / hidden_layer_2_size)
        self.bias3 = np.zeros((output_layer_size, 1))

    def relu(self, x):
      return np.maximum(0, x)#calculating the max between each element in x and 0. Returns array of 0's 
    #and numbers greater than 0

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def forward_propagation(self, x):
        z1 = self.Weight1 @ x + self.bias1
        a1 = self.relu(z1)
        z2 = self.Weight2 @ a1 + self.bias2
        a2 = self.relu(z2)
        z3 = self.Weight3 @ a2 + self.bias3
        a3 = self.softmax(z3)
        return a1, a2, a3

    def backward_propagation(self, x, y_true, a1, a2, a3):
        y = np.zeros_like(a3)#array of all zeros in the shape of a3
        y[y_true] = 1 #make the right spot (as in the right answer) in y equal to 1. Ex: if the ans should be 3, y[3]=1
        deriv_z3 = a3 - y
        deriv_W3 = deriv_z3 @ a2.T
        deriv_b3 = deriv_z3
        deriv_z2 = (self.Weight3.T @ deriv_z3) * self.relu_derivative(a2)
        deriv_W2 = deriv_z2 @ a1.T
        deriv_b2 = deriv_z2
        deriv_z1 = (self.Weight2.T @ deriv_z2) * self.relu_derivative(a1)
        deriv_W1 = deriv_z1 @ x.T
        deriv_b1 = deriv_z1

        self.Weight3 -= self.learning_rate * deriv_W3
        self.bias3 -= self.learning_rate * deriv_b3
        self.Weight2 -= self.learning_rate * deriv_W2
        self.bias2 -= self.learning_rate * deriv_b2
        self.Weight1 -= self.learning_rate * deriv_W1
        self.bias1 -= self.learning_rate * deriv_b1

    def predict(self, X):
        predictions = []
        for x in X:
            _, _, output = self.forward_propagation(x.reshape(-1, 1))
            predictions.append(np.argmax(output))
        return predictions

    def train(self, X, y, X_val, y_val, epochs=5):
        best_model = copy.deepcopy(self)
        best_val_acc = 0
        for _ in range(epochs):
            for i in range(len(X)):
                x = X[i].reshape(-1, 1)
                a1, a2, a3 = self.forward_propagation(x)
                self.backward_propagation(x, y[i], a1, a2, a3)
            val_acc = evaluate(self, X_val, y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(self)
        return best_model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return np.mean(np.array(preds) == y_test)

def run_experiment(train_img_data, train_label_data, validation_img_data, validation_label_data, test_img_data, test_lbl_data, num_classes, height, width, label="dataset"):

    x_train_full, y_train_full = load_data(train_img_data, train_label_data, height, width)
    x_val, y_val = load_data(validation_img_data, validation_label_data, height, width)
    x_test, y_test = load_data(test_img_data, test_lbl_data, height, width)
    input_size = height * width

    print(f"\n{label}:")
    for percent_data in range(10, 101, 10):
        accuracies = []
        start_time = time.time()

        for _ in range(5):  # 5 runs
            indices = list(range(len(x_train_full)))
            random.shuffle(indices)
            subset_size = int(percent_data / 100 * len(indices))
            selected = indices[:subset_size]

            X_train = x_train_full[selected]
            y_train = y_train_full[selected]

            model = NeuralNet(input_size, 128, 64, num_classes, learning_rate=0.01)
            best_model = model.train(X_train, y_train, x_val, y_val, epochs=5)
            accuracy = evaluate(best_model, x_test, y_test)
            accuracies.append(accuracy)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        elapsed = time.time() - start_time
        print(f"{percent_data}% data => Test Acc: {mean_acc:.4f} ± {std_acc:.4f}, Time: {elapsed:.2f}s")

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