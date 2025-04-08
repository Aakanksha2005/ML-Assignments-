# Question 2: Logistic Regression, MLP, CNN & PCA
# %%
#Dataset loading for question 2
import Oracle_Assignment_2
from Oracle_Assignment_2 import q2_get_mnist_jpg_subset

import zipfile
import os

data2 = q2_get_mnist_jpg_subset(23647)

extract_folder = f"q2_data"

# %%
#Loading Data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import torch 
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


data2 = ImageFolder(extract_folder, transform=transform)

class_wise_data = {i: [] for i in range(10)}  

for path, label in data2.imgs:
    class_wise_data[label].append((path, label))

train_imgs = []
test_imgs = []

for label, images in class_wise_data.items():
    random.shuffle(images)
    split_point = int(len(images)*0.8)
    train_imgs.extend(images[:split_point])
    test_imgs.extend(images[split_point:])

train_data_2 = ImageFolder(extract_folder, transform=transform)
train_data_2.imgs = train_imgs
train_data_2.samples = train_imgs

test_data_2 = ImageFolder(extract_folder, transform=transform)
test_data_2.imgs = test_imgs
test_data_2.samples = test_imgs 

train_load_2 = DataLoader(train_data_2, batch_size=64, shuffle=True)
test_load_2 = DataLoader(test_data_2, batch_size=64, shuffle=False)


# %%
#Checking the dataset
import matplotlib.pyplot as plt
import numpy as np

# Display a few images
fig, ax = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    img, label = data2[i+7200]
    ax[i].imshow(img[0], cmap='gray')
    ax[i].set_title(f'Label: {label}')
    ax[i].axis('off')
    
plt.show()


# %%
#MLP model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train(model, train_loader, epochs, lr, momentum, criterion):
    model.train()
    vel = {p: torch.zeros_like(p) for p in model.parameters()}  
    
    for e in range(epochs):
        epoch_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to('cpu'), labels.to('cpu')
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    vel[p] = momentum * vel[p] + lr * p.grad 
                    p.data -= lr * vel[p]
            for p in model.parameters():
                p.grad = None 
            
            epoch_loss += loss.item()
        
        print(f"Epoch: {e+1}, Loss: {epoch_loss/len(train_loader)}")
    
    return model

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to('cpu'), labels.to('cpu')
            op = model(imgs)
            _, predicted = torch.max(op, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    return correct / total, conf_matrix

model_mlp = MLP()
criterion_mlp = nn.CrossEntropyLoss()

model_mlp = train(model_mlp, train_load_2, epochs=100, lr=0.01, momentum=0.997, criterion=criterion_mlp)

test_accuracy_mlp, conf_matrix_mlp = test(model_mlp, test_load_2)
print(f"Test Accuracy of MLP: {test_accuracy_mlp*100:.4f}%")
print("Confusion Matrix:")
print(conf_matrix_mlp)

train_accuracy_mlp, _ = test(model_mlp, train_load_2)
print(f"Train Accuracy: {train_accuracy_mlp*100:.4f}%")

true_pos_mlp = np.diag(conf_matrix_mlp)
false_pos_mlp = np.sum(conf_matrix_mlp, axis=0) - true_pos_mlp
false_neg_mlp = np.sum(conf_matrix_mlp, axis=1) - true_pos_mlp
true_neg_mlp = np.sum(conf_matrix_mlp) - (true_pos_mlp + false_pos_mlp + false_neg_mlp)

precision_mlp = true_pos_mlp / (true_pos_mlp + false_pos_mlp)
recall_mlp = true_pos_mlp / (true_pos_mlp + false_neg_mlp)
f1_mlp = 2 * precision_mlp * recall_mlp / (precision_mlp + recall_mlp)

print(f"Precision_mlp: {precision_mlp}")
print(f"Recall_mlp: {recall_mlp}")
print(f"F1_mlp: {f1_mlp}")
