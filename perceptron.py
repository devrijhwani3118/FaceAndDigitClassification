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

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size + 1)
        self.learning_rate = learning_rate
        
    def step_function(self, x):
        return (x > 0).astype(float)

    def score(self, X):
        X = np.c_[X, np.ones((X.shape[0]))]
        return X @ self.weights

    def predict(self, X):
        X = np.c_[X, np.ones((X.shape[0]))]
        preds = np.zeros(len(X))
        for i in range(len(X)):
            preds[i] = self.step_function(np.dot(X[i], self.weights))
        return preds

    def train(self, X, y, epochs=5):
        X = np.c_[X, np.ones((X.shape[0]))]
        for _ in range(epochs):
            for i in range(len(X)):
                pred = self.step_function(np.dot(X[i], self.weights))
                if pred != y[i]:
                    self.weights += -1 * self.learning_rate * X[i] * (pred - y[i])
        return copy.deepcopy(self)

class MultPerceptron:
    def __init__(self, num_classes, input_size, learning_rate=0.01):
        self.num_classes = num_classes
        self.classifiers = [Perceptron(input_size, learning_rate) for _ in range(num_classes)]

    def train(self, X, y, epochs=5):
        for c in range(self.num_classes):
            binary_labels = (y == c).astype(float)
            self.classifiers[c] = self.classifiers[c].train(X, binary_labels, epochs)
        return copy.deepcopy(self)

    def predict(self, X):
        scores = np.stack([clf.score(X) for clf in self.classifiers], axis=1)
        return np.argmax(scores, axis=1)

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

            model = MultPerceptron(num_classes, input_size, learning_rate=0.01)
            best_model = model.train(X_train, y_train, epochs=5)
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