import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision.models as models

import torch.nn as nn
import torch.optim as optim
# Set the paths to the abnormal and normal folders
folder = r'YOUR_PATH\Brain_tumor_detection\dataset'

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = ImageFolder(root=folder, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load the pre-trained alexnet model
alexnet = models.alexnet(pretrained=True)

# Freeze the weights of the pre-trained layers
for param in alexnet.parameters():
    param.requires_grad = False

# Modify the last fully connected layer for binary classification
num_features = alexnet.classifier[6].in_features
alexnet.classifier[6] = nn.Linear(num_features, 2)
# Add class labels
alexnet.class_labels = ['benign', 'malignant']


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.classifier.parameters(), lr=0.001, momentum=0.9)

# Train the model
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet = alexnet.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = alexnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}")

    # Save the trained model
    torch.save(alexnet.state_dict(), 'trained_model.pth')
