# py -u "c:\Users\dbotn\VSCode\FaceAndDigitClassification\neural_net_pytorch.py"

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class CustomDataset(Dataset):
    def __init__(self, img_file, label_file, transform = None):
        with open(img_file, 'r') as f:
            lines = f.readlines()
        
        self.images = []
        current_image = []
        separator = ' ' * 28

        for line in lines:
            line = line.rstrip('\n')
            if line == separator:
                if current_image:
                    self.images.append('\n'.join(current_image))
                    current_image = []
            else:
                current_image.append(line)

        if current_image:
            self.images.append('\n'.join(current_image))
        
        with open(label_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f]
        
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        ascii = self.images[i]
        img_array = self._ascii_to_array(ascii)
        
        label = self.labels[i]
        
        if self.transform:
            img_array = self.transform(img_array)
            
        return img_array, label
    
    def _ascii_to_array(self, ascii):
        lines = [line for line in ascii.split('\n') if line.strip()]

        img_array = np.zeros((28, 28), dtype = np.float32)

        for i, line in enumerate (lines):
            for j, char in enumerate(line):
                if i < 28 and j < 28:
                    if char == '#' or char == '+':
                        img_array[i, j] = 1
                    elif char == ' ':
                        img_array[i, j] = 0
        return img_array

transform = torch.from_numpy
train_data = CustomDataset(
    img_file = "data/digitdata/trainingimages",
    label_file = "data/digitdata/traininglabels",
    transform = transform
)
test_data = CustomDataset(
    img_file = "data/digitdata/testimages",
    label_file = "data/digitdata/testlabels",
    transform = transform
)

train_dataloader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")