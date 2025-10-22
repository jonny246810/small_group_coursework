import torch
import torch.nn as nn

class PlantCNN(nn.Module):
    def __init__(self, num_classes=39):
        super(PlantCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # 256x256x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 256x256x32
        self.pool1 = nn.MaxPool2d(2, 2) #128x128x32
        
        # Block 2: 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #128x128x64
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1) #128x128x128
        self.pool2 = nn.MaxPool2d(2, 2)   #64x64x128
        
       #dense layers
        self.fc1 = nn.Linear(128 * 64 * 64, 1024) #1024 neurons
        self.fc2 = nn.Linear(1024, 512) #512 neurons
        self.fc3 = nn.Linear(512, num_classes) #39 neurons
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Block 1
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Block 2
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Return logits 
        
        return x
    
if __name__ == "__main__":
    model = PlantCNN(num_classes=39)
    
    dummy_input = torch.randn(4, 3, 256, 256)
    output = model(dummy_input)
    
    print(f"Model output shape: {output.shape}")  # Should be [4, 39]
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")