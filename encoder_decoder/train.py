import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import optim
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import random
import string

def generate_random_text(length):
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for _ in range(length))

class ImageDataset(Dataset):
    def __init__(self, n=200, size=128, mode=1):
        super().__init__()
        self.n = n
        self.size = size
        self.mode = mode
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = Image.new("L",(self.size, self.size), color=255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        if self.mode == 1:
            text = "ABC"
            x = np.random.randint(10,self.size-40)
            y = np.random.randint(10,self.size-40)

        elif self.mode == 2:
            text = generate_random_text(3)
            x = 30
            y = 30

        elif self.mode == 3:
            length = random.randint(2,5)
            text = generate_random_text(length)
            x = 30
            y = 30

        elif self.mode == 4:
            length = random.randint(2, 5)
            text = generate_random_text(length)
            x = np.random.randint(10, self.size - 40)
            y = np.random.randint(10, self.size - 40)

        draw.text((x, y), text, fill=0, font=font)
        tensor = self.transforms(image)
        return tensor, tensor

class Encoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.bottleneck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.bottleneck = nn.Linear(latent, 256 * 16 * 16)

        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x


def run_experiment(mode, epochs=10):
    print(f"\n{mode}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageDataset(2000, 256, mode=mode)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

    encoder.train()
    decoder.train()

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            latent = encoder(imgs)
            output = decoder(latent)
            loss = criterion(imgs, output)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        final_loss = avg_loss
        print(f"Эпоха {epoch + 1}/{epochs}, Loss: {avg_loss:.5f}")

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        test_dataset = ImageDataset(10, 256, mode=mode)
        image, _ = test_dataset[0]
        image_gpu = image.unsqueeze(0).to(device)

        latent = encoder(image_gpu)
        result = decoder(latent).cpu()

    plt.figure(figsize=(12, 4))
    plt.suptitle(f"{mode} Loss: {final_loss:.5f}")

    plt.subplot(131)
    plt.imshow(image.squeeze(), cmap='gray')

    plt.subplot(132)
    plt.imshow(result.squeeze(), cmap='gray')

    plt.subplot(133)
    plt.imshow((image.squeeze() - result.squeeze()).abs(), cmap='gray')

    plt.show()
    return final_loss

if __name__ == '__main__':
    results = {}

    for m in range(1, 5):
        loss = run_experiment(mode=m, epochs=10)
        results[f"{m}"] = loss

    for mode_name, loss_val in results.items():
        print(f"{mode_name}: {loss_val:.5f}")
