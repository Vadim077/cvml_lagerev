import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import time

save_path = Path(__file__).parent
zip_path = save_path / "cyrillic.zip"
data_dir = save_path / "Cyrillic"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CyrillicDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = str(self.paths[idx])

        with Image.open(img_path) as img:
            image = img.getchannel('A')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image,label

class CyrillicCNN(nn.Module):
    def __init__(self):
        super(CyrillicCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*8*8, 256)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256,34)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__=='__main__':

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)

    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    all_paths = []
    all_labels = []

    for cls_name in classes:
        cls_dir = data_dir / cls_name
        for img_path in cls_dir.glob("*.png"):
            all_paths.append(img_path)
            all_labels.append(class_to_idx[cls_name])

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CyrillicDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = CyrillicDataset(test_paths, test_labels, transform=test_transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Batch: train: {len(train_loader)}, test: {len(test_loader)}")

    model = CyrillicCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 9
    train_losses = []
    test_accuraci = []

    model_path = save_path / "cyrillic_model.pth"

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_acc = 100 * (correct/total)
        test_accuraci.append(epoch_acc)
        elapsed_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs}| time: {elapsed_time} | mistake: {epoch_loss} | acc: {epoch_acc}")

    torch.save(model.state_dict(), model_path)
    print("Model save")

    plt.figure(figsize=(12,5))

    plt.subplot(121)
    plt.title("Loss")
    plt.plot(train_losses, color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Mistake")

    plt.subplot(1,2,2)
    plt.title("Accuracy")
    plt.plot(test_accuraci, color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Acc(%)")

    plt.savefig(save_path/"train.png")
    plt.show()
