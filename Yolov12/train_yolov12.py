import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from yolov12_model import YOLOv12

# Custom dataset for YOLO format
def get_image_paths_and_labels(data_dir, class_names):
    image_paths = []
    labels = []
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(idx)
    return image_paths, labels

class SimpleCoalDataset(Dataset):
    def __init__(self, data_dir, class_names, transform=None):
        self.image_paths, self.labels = get_image_paths_and_labels(data_dir, class_names)
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def train():
    class_names = [
        "destructive_coal",
        "fully_pulverized_coal",
        "non_destructive_coal",
        "not_coal",
        "pulverized_coal",
        "strongly_destructive_coal"
    ]
    data_dir = 'data'
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])
    dataset = SimpleCoalDataset(data_dir, class_names, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    model = YOLOv12(num_classes=len(class_names)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            # For demonstration, use global average pooling and cross-entropy
            pooled = outputs.mean([2, 3])
            loss = criterion(pooled, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), 'yolov12_coal.pth')
    print("Training complete. Model saved as yolov12_coal.pth")

if __name__ == "__main__":
    train()
