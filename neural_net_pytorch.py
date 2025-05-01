# py -u "c:\Users\dbotn\VSCode\FaceAndDigitClassification\neural_net_pytorch.py"

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import time
import random

PIXEL_VALUES = {' ': 0, '+': 1, '#': 1}

def load_data(image_file, label_file, height, width):
    labels = []
    lines = []
    flattened_image = []
    
    with open(label_file, 'r') as f:
        for line in f:
            if line.strip():
                clean_line = line.strip()
                labels.append(int(clean_line))
    
    with open(image_file, 'r') as f:
        for line in f:
            img_line = line.rstrip('\n')
            lines.append(img_line)
    
    num_images = len(labels)
    
    images_tensor = torch.zeros((num_images, height * width), dtype=torch.float32)
    
    for i in range(num_images):
        start = i * height
        end = start + height
        image_lines = lines[start:end]
        
        for h, line in enumerate(image_lines):
            row_start = h * width
            for w, c in enumerate(line.ljust(width)[:width]):
                value = PIXEL_VALUES.get(c, 0)
                images_tensor[i, row_start + w] = value
    
    return images_tensor, torch.tensor(labels, dtype=torch.long)

class NeuralNet:
    def __init__(self, input_size, hidden_layer_1_size, hidden_layer_2_size, 
                 output_layer_size, learning_rate=0.01, device='cpu'):
        self.learning_rate = learning_rate
        self.device = device
        
        scale1 = torch.sqrt(torch.tensor(1. / input_size)).to(device)
        self.Weight1 = nn.Parameter(torch.randn(hidden_layer_1_size, input_size, device=device) * scale1)
        self.bias1 = nn.Parameter(torch.zeros(hidden_layer_1_size, 1, device=device))
        
        scale2 = torch.sqrt(torch.tensor(1. / hidden_layer_1_size)).to(device)
        self.Weight2 = nn.Parameter(torch.randn(hidden_layer_2_size, hidden_layer_1_size, device=device) * scale2)
        self.bias2 = nn.Parameter(torch.zeros(hidden_layer_2_size, 1, device=device))
        
        scale3 = torch.sqrt(torch.tensor(1. / hidden_layer_2_size)).to(device)
        self.Weight3 = nn.Parameter(torch.randn(output_layer_size, hidden_layer_2_size, device=device) * scale3)
        self.bias3 = nn.Parameter(torch.zeros(output_layer_size, 1, device=device))
        
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

    def parameters(self):
        return [self.Weight1, self.bias1, self.Weight2, self.bias2, self.Weight3, self.bias3]

    def relu(self, x):
        return F.relu(x)

    def relu_derivative(self, x):
        return (x > 0).float()

    def softmax(self, x):
        return F.softmax(x, dim=0)

    def forward_propagation(self, x):
        x = x.to(self.device)
        z1 = self.Weight1 @ x + self.bias1
        a1 = self.relu(z1)
        z2 = self.Weight2 @ a1 + self.bias2
        a2 = self.relu(z2)
        z3 = self.Weight3 @ a2 + self.bias3
        a3 = self.softmax(z3)
        return a1, a2, a3

    def backward_propagation(self, x, y_true, a1, a2, a3):
        self.optimizer.zero_grad()
        y = torch.zeros_like(a3)
        y[y_true] = 1
        loss = -torch.sum(y * torch.log(a3))
        loss.backward()
        self.optimizer.step()

    def predict(self, X):
        predictions = []
        with torch.no_grad():
            for x in X:
                x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).view(-1, 1)
                _, _, output = self.forward_propagation(x_tensor)
                predictions.append(torch.argmax(output).item())
        return predictions

    def train(self, X, y, X_val, y_val, epochs=5):
        best_model = copy.deepcopy(self)
        best_val_acc = 0
        
        for _ in range(epochs):
            for i in range(len(X)):
                x = torch.tensor(X[i], dtype=torch.float32, device=self.device).view(-1, 1)
                a1, a2, a3 = self.forward_propagation(x)
                self.backward_propagation(x, y[i], a1, a2, a3)
            
            val_acc = evaluate(self, X_val, y_val)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(self)
                
        return best_model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.long)
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds, dtype=torch.long)
    return (preds == y_test).float().mean().item()

def run_experiment(train_img_data, train_label_data, validation_img_data, 
                  validation_label_data, test_img_data, test_lbl_data, 
                  num_classes, height, width, label="dataset", device='cpu'):

    x_train_full, y_train_full = load_data(train_img_data, train_label_data, height, width)
    x_val, y_val = load_data(validation_img_data, validation_label_data, height, width)
    x_test, y_test = load_data(test_img_data, test_lbl_data, height, width)
    
    x_train_full = x_train_full.to(device)
    y_train_full = y_train_full.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    input_size = height * width

    print(f"\n{label}:")
    for percent_data in range(10, 101, 10):
        accuracies = []
        start_time = time.time()

        for _ in range(5): 
            indices = list(range(len(x_train_full)))
            random.shuffle(indices)
            subset_size = int(percent_data / 100 * len(indices))
            selected = indices[:subset_size]

            X_train = x_train_full[selected]
            y_train = y_train_full[selected]

            model = NeuralNet(input_size, 128, 64, num_classes, learning_rate=0.01, device=device)
            best_model = model.train(X_train, y_train, x_val, y_val, epochs=5)
            accuracy = evaluate(best_model, x_test, y_test)
            accuracies.append(accuracy)

        accuracies = torch.tensor(accuracies)
        mean_acc = accuracies.mean().item()
        std_acc = accuracies.std().item()
        elapsed = time.time() - start_time
        print(f"{percent_data}% data => Test Acc: {mean_acc:.4f} ± {std_acc:.4f}, Time: {elapsed:.2f}s")

# ---- Main ----
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    run_experiment(
        "data/digitdata/trainingimages", "data/digitdata/traininglabels",
        "data/digitdata/validationimages", "data/digitdata/validationlabels",
        "data/digitdata/testimages", "data/digitdata/testlabels",
        num_classes=10, height=28, width=28, label="Digit Classification",
        device=device
    )

    run_experiment(
        "data/facedata/facedatatrain", "data/facedata/facedatatrainlabels",
        "data/facedata/facedatavalidation", "data/facedata/facedatavalidationlabels",
        "data/facedata/facedatatest", "data/facedata/facedatatestlabels",
        num_classes=2, height=70, width=60, label="Face Classification",
        device=device
    )