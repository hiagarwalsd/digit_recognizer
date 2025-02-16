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
from sklearn.model_selection import train_test_split
import import_ipynb
from Model import DigitCNN


# In[2]:


class MNISTDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.long)
        self.images = torch.tensor(data.iloc[:, 1:].values.reshape(-1, 1, 28, 28), dtype=torch.float32) / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# In[3]:


dataset = MNISTDataset('mnist_train.csv')
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


# In[4]:


best_val_acc = 0.0  # Track the best validation accuracy
epoch_losses = []
val_accuracies = []  # Store validation accuracies

epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_losses.append(running_loss)

    # Evaluate on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    # Save the model if the validation accuracy improves
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")
        print("New best model saved!")

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Val Acc: {val_accuracy:.4f}")


# In[5]:


# Subplot 1: Epoch vs Loss
epochs = list(range(1, 16))
plt.subplot(1, 2, 1)
plt.plot(epochs, epoch_losses, marker='o', color='b', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.legend()
plt.grid(True)

# Subplot 2: Epoch vs Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, marker='o', linestyle='-', color='g', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Epoch vs Validation Accuracy')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# In[6]:


model.load_state_dict(torch.load('best_model.pth'))
model.eval()
correct = 0
total = 0
misclassified_images = []
misclassified_labels = []
predicted_labels = []
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in val_loader:
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


# In[7]:


fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for i in range(min(6, len(misclassified_images))):
    ax = axes[i]
    ax.imshow(misclassified_images[i].squeeze(), cmap='gray')
    ax.set_title(f'True: {misclassified_labels[i].item()}\nPred: {predicted_labels[i].item()}')
    ax.axis('off')
plt.show()


# In[8]:


conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[9]:


print("Classification Report:")
print(classification_report(all_labels, all_predictions))

precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


# In[ ]:




