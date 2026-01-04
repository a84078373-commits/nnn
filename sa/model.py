import torch
import torch.nn as nn
from torchvision import models

class ASLResNet(nn.Module):
    def __init__(self, num_classes):
        super(ASLResNet, self).__init__()
        
        # Load pre-trained ResNet-50 (using weights for newer torchvision)
        try:
            self.model = models.resnet50(weights='IMAGENET1K_V1')
        except:
            try:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            except:
                self.model = models.resnet50(pretrained=True)
        
        # Replace classifier head
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
