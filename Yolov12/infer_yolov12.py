import torch
from torchvision import transforms
from PIL import Image
from yolov12_model import YOLOv12

# List of class names
class_names = [
    "destructive_coal",
    "fully_pulverized_coal",
    "non_destructive_coal",
    "not_coal",
    "pulverized_coal",
    "strongly_destructive_coal"
]

def predict(image_path, model_path='yolov12_coal.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv12(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pooled = output.mean([2, 3])  # Global average pooling
        pred = torch.argmax(pooled, dim=1).item()
        class_name = class_names[pred]
    print(f"Predicted class: {class_name}")
    return class_name

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python infer_yolov12.py <image_path>")
    else:
        predict(sys.argv[1])
