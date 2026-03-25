import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import random
from train_model import CyrillicCNN

save_path = Path(__file__).parent
data_dir = save_path / "Cyrillic"
model_path = save_path / "cyrillic_model.pth"

classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CyrillicCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

all_images = list(data_dir.rglob("*.png"))
random_sampl = random.sample(all_images, 9)
plt.figure(figsize=(10,10))

for i, img_path in enumerate(random_sampl):
    true_label = img_path.parent.name

    with Image.open(img_path) as img:
        alpha = img.getchannel('A')
        img_tensor = transform(alpha).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred_idx = torch.max(output,1)
        pred_label = classes[pred_idx.item()]

    plt.subplot(3,3,i+1)
    plt.imshow(alpha, cmap='gray')
    color = 'green' if true_label == pred_label else 'red'

    plt.title(f"True {true_label}\nPred {pred_label}", color=color, fontsize=14)
    plt.axis('off')

plt.tight_layout()
plt.show()
