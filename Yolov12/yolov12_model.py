import torch
import torch.nn as nn

# This is a simplified YOLOv12-like model for demonstration. For real use, replace with official YOLOv12 code if available.
class YOLOv12(nn.Module):
    def __init__(self, num_classes=6):
        super(YOLOv12, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, (num_classes + 5) * 3, 1)  # 3 anchors
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    model = YOLOv12(num_classes=6)
    x = torch.randn(1, 3, 416, 416)
    y = model(x)
    print(y.shape)  # Should be (1, (num_classes+5)*3, 52, 52)
