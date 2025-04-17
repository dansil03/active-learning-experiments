import torch.nn as nn
from torchvision.models import resnet18

class MCResNet18(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.5):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        
        # Voeg dropout toe vóór de laatste classifier
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


