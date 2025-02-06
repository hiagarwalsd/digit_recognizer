#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import import_ipynb
from Model import DigitCNN


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN().to(device)

state_dict = torch.load('best_model.pth', map_location=device)

# Rename keys to match the new Sequential model
new_state_dict = {}
for key in state_dict.keys():
    if "conv1" in key:
        new_state_dict[key.replace("conv1", "features.0")] = state_dict[key]
    elif "conv2" in key:
        new_state_dict[key.replace("conv2", "features.3")] = state_dict[key]
    elif "conv3" in key:
        new_state_dict[key.replace("conv3", "features.6")] = state_dict[key]
    elif "fc1" in key:
        new_state_dict[key.replace("fc1", "classifier.0")] = state_dict[key]
    elif "fc2" in key:
        new_state_dict[key.replace("fc2", "classifier.3")] = state_dict[key]
    else:
        new_state_dict[key] = state_dict[key]

# Load into the new model
model = DigitCNN().to(device)
model.load_state_dict(new_state_dict)


# In[3]:


class MNISTDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.long)
        self.images = torch.tensor(data.iloc[:, 1:].values.reshape(-1, 1, 28, 28), dtype=torch.float32) / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
dataset = MNISTDataset('mnist_test.csv')
test_size = len(dataset)



test_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
model.eval()
correct = 0
total = 0
misclassified_images = []
misclassified_labels = []
predicted_labels = []
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        misclassified_idx = (predicted != labels).nonzero(as_tuple=True)[0]
        misclassified_images.extend(images[misclassified_idx].cpu())
        misclassified_labels.extend(labels[misclassified_idx].cpu())
        predicted_labels.extend(predicted[misclassified_idx].cpu())
        
        all_labels.extend(labels.cpu())
        all_predictions.extend(predicted.cpu())

print(f'Accuracy: {100 * correct / total:.2f}%')


# In[4]:


fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for i in range(min(6, len(misclassified_images))):
    ax = axes[i]
    ax.imshow(misclassified_images[i].squeeze(), cmap='gray')
    ax.set_title(f'True: {misclassified_labels[i].item()}\nPred: {predicted_labels[i].item()}')
    ax.axis('off')
plt.show()


# In[5]:


conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[6]:


print("Classification Report:")
print(classification_report(all_labels, all_predictions))

precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

