import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import os 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#######################################################################
#Read dataset
df_train = pd.read_csv("dataset/train.csv", skiprows=1, header=None, names=["Index", "Image_Path", "Label"])
df_train = df_train.drop(columns=["Index"])

df_test = pd.read_csv("dataset/test.csv", skiprows=1, header=None, names=["Index", "Image_Path", "Label"])
df_test = df_test.drop(columns=["Index"])


transform = transforms.Compose([
    transforms.Resize((224, 224)), #Resize
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) #Normalize
])

class ImageDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe #Store dataset
        self.root_dir = root_dir # Root directory
        self.transform = transform # image transformation

    def __len__(self):
        return len(self.dataframe) #size of dataset (returns the number of rows in dataset)

    # retrieve a single image and associated label
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        label = int(self.dataframe.iloc[idx, 1])

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
    
# Dataset objects
train_dataset = ImageDataset(df_train, root_dir="dataset/", transform=transform)
test_dataset = ImageDataset(df_test, root_dir="dataset/", transform=transform)

# DataLoaders for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class AIImageClassifier(nn.Module):
    # Layers definition
    def __init__(self):
        super(AIImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 112 * 112, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
    
# Initialize model
model = AIImageClassifier()

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 64
# train_dataset = TensorDataset(image, label)
# train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)

#######################################################################
# Training
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0.0

    for image, label in train_loader:
        image, label = image.to(device), label.float().to(device)

        optimizer.zero_grad()
        outputs = model(image).squeeze()  # Squeeze to match label shape
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

print("Training complete!")


# Test model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.float().to(device)
        outputs = model(images).squeeze()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {100 * correct / total:.2f}%")

#######################################################################
# Define new images
def predict_image(image_path, model):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    output = model(image).item()
    prediction = "AI-generated" if output > 0.5 else "Human-made"
    print(f"Prediction: {prediction} (Confidence: {output:.2f})")

predict_image("train_data/sample.jpg", model)

    

