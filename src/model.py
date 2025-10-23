import torch
import torch.nn as nn

class PlantCNN(nn.Module):
    def __init__(self, num_classes=39):
        super(PlantCNN, self).__init__()
        
        self.conv1_1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.relu(x)
        
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.relu(x)
        
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = self.relu(x)
        
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.pool4(x)
        x = self.relu(x)
        
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.pool5(x)
        x = self.relu(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
if __name__ == "__main__":
    model = PlantCNN(num_classes=39)
    
    dummy_input = torch.randn(4, 3, 256, 256)
    output = model(dummy_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")