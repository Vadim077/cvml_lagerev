import torch
import torch.nn as nn
import cv2
import torchvision
import numpy as np
from torchvision import transforms
import time
from collections import deque
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(model_type):
    if model_type == "efficientnet":
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = torchvision.models.efficientnet_b0(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False
        features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(features, 1)

    elif model_type == "alexnet":
        weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        model = torchvision.models.alexnet(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False
        features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(features, 1)
    return model.to(device)


path_eff = "effnet.pth"
path_alex = "alexnet.pth"

model_eff = build_model("efficientnet")
if os.path.exists(path_eff):
    model_eff.load_state_dict(torch.load(path_eff, map_location=device, weights_only=True))

model_alex = build_model("alexnet")
if os.path.exists(path_alex):
    model_alex.load_state_dict(torch.load(path_alex, map_location=device, weights_only=True))
print("ready")

criterion = nn.BCEWithLogitsLoss()

optimizer_eff = torch.optim.Adam(filter(lambda p: p.requires_grad, model_eff.parameters()), lr=0.001)
optimizer_alex = torch.optim.Adam(filter(lambda p: p.requires_grad, model_alex.parameters()), lr=0.001)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class Buffer():
    def __init__(self, mazsize=16):
        self.frames = deque(maxlen=mazsize)
        self.labels = deque(maxlen=mazsize)

    def append(self, tensor, label):
        self.frames.append(tensor)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)

    def get_batch(self):
        images = torch.stack(list(self.frames)).to(device)
        labels = torch.tensor(list(self.labels), dtype=torch.float32).to(device)
        return images, labels

def train(model, optimizer, buffer):
    if len(buffer) < 10:
        return None
    model.train()

    images, labels = buffer.get_batch()

    final_loss = 0
    for _ in range(3):
        optimizer.zero_grad()
        predictions = model(images).squeeze()

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    return final_loss

def predict(model, frame):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()
    label = "person" if prob > 0.5 else "no_person"
    return label, prob

cap = cv2.VideoCapture(0)

cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
buffer = Buffer()
count_labeled = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if key == ord("q"):
        break
    elif key == ord("1"):  # person
        tensor = transform(image)
        buffer.append(tensor, 1.0)
        count_labeled += 1
        print(f"Добавлен: {len(buffer)}/16")
    elif key == ord("2"):  # no person
        tensor = transform(image)
        buffer.append(tensor, 0.0)
        count_labeled += 1
        print(f"Добавлен: {len(buffer)}/16")

    elif key == ord("p"):  # predict
        t1 = time.perf_counter()
        label_eff, conf_eff = predict(model_eff, frame)
        t_eff = time.perf_counter() - t1

        t2 = time.perf_counter()
        label_alex, conf_alex = predict(model_alex, frame)
        t_alex = time.perf_counter() - t2

        print("\n итог")
        print(f"EfficientNet : {label_eff:10} | Уверенность: {conf_eff:.3f} | Время: {t_eff:.4f}s")
        print(f"AlexNet      : {label_alex:10} | Уверенность: {conf_alex:.3f} | Время: {t_alex:.4f}s")

    elif key == ord("s"):  # save model
        torch.save(model_eff.state_dict(), path_eff)
        torch.save(model_alex.state_dict(), path_alex)
        print("Модели сохранены")

    if count_labeled >= buffer.frames.maxlen:
        print("\n обучение")
        loss_eff = train(model_eff, optimizer_eff, buffer)
        loss_alex = train(model_alex, optimizer_alex, buffer)

        print(f"EfficientNet Loss: {loss_eff:.4f}")
        print(f"AlexNet Loss     : {loss_alex:.4f}")

        count_labeled = 0

cap.release()
cv2.destroyAllWindows()