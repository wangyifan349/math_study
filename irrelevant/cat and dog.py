# Required libraries:
# pip install torch torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Device configuration: use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations for training dataset
# Includes resizing, random horizontal flipping for augmentation, tensor conversion, and normalization
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          # Resize to 224x224 pixels (required input size for ResNet)
    transforms.RandomHorizontalFlip(),      # Randomly flip image horizontally to augment training data
    transforms.ToTensor(),                  # Convert images to PyTorch tensors (scales pixel values to [0,1])
    transforms.Normalize(                   # Normalize using ImageNet dataset mean and std for each channel (RGB)
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225])
])

# Define image transformations for validation dataset
# Resizing and normalization only, no data augmentation applied
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),          # Resize to 224x224 pixels
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(                    # Normalize using ImageNet mean and std
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225])
])

# Data directory - replace with your actual dataset directory path
# The directory must contain 'train' and 'val' subfolders, each with 'cat' and 'dog' folders inside
data_dir = 'data'  # root directory of the dataset
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Load the training dataset using ImageFolder which reads folders as class labels automatically
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)

# Load the validation dataset similarly
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

# Data loaders provide batches of images and labels during training and validation
# num_workers sets how many subprocesses to use for data loading; adjust depending on your OS/CPU cores
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load pretrained ResNet-18 model - pretrained on ImageNet dataset
model = models.resnet18(pretrained=True)

# Replace the final fully connected layer to classify only 2 classes: cat and dog
num_features = model.fc.in_features         # Number of input features to the last layer
model.fc = nn.Linear(num_features, 2)       # Change output layer to have 2 outputs

# Transfer model to GPU or CPU
model = model.to(device)

# Define the loss function (CrossEntropyLoss is suitable for multi-class classification)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (Adam optimizer with learning rate of 0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to train the model for one epoch
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Set model to training mode (enables dropout, batchnorm updates, etc.)
    running_loss = 0.0           # Total loss accumulated over epoch
    running_corrects = 0         # Total correct predictions over epoch
    total_samples = 0            # Total samples processed in epoch

    # Iterate over the data loader batches
    for inputs, labels in dataloader:
        inputs = inputs.to(device)      # Move inputs to device
        labels = labels.to(device)      # Move labels to device

        optimizer.zero_grad()           # Clear previous gradients

        outputs = model(inputs)         # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        loss.backward()                 # Backward pass (compute gradients)
        optimizer.step()                # Update model parameters

        # Update running loss and correct predictions count
        running_loss += loss.item() * inputs.size(0)  # Multiply by batch size
        _, preds = torch.max(outputs, 1)               # Get class predictions
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples       # Compute average loss
    epoch_acc = running_corrects / total_samples    # Compute accuracy

    return epoch_loss, epoch_acc

# Function to evaluate the model on validation dataset
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()       # Set model to evaluation mode (disables dropout, batchnorm update)
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for inference
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)          # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc

# Number of epochs to train
num_epochs = 10

# Training and validation loop
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

# Save the trained model weights for later use or deployment
torch.save(model.state_dict(), "cat_dog_resnet18.pth")
print("Model weights saved to cat_dog_resnet18.pth")
