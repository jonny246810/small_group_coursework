import torch
import torch.nn as nn

class PlantCNN(nn.Module):
    def __init__(self, num_classes=39):
        super(PlantCNN, self).__init__()
        
        # Block 1: 256x256x3 to 128x128x8
        self.conv1_1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 128x128x8 to 64x64x16
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: 64x64x16 to 32x32x32
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4: 32x32x32 to 16x16x64
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Block 5: 16x16x64 to 8x8x128
        self.conv5_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # Single dense layer: 8x8x128 to 39 neuron
        self.fc = nn.Linear(8 * 8 * 128, num_classes) #GAP - average go 1x1x128. maybe 1x1x256
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Block 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)
        
        # Block 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)
        
        # Block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.pool3(x)
        
        # Block 4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.pool4(x)
        
        # Block 5
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.pool5(x)
        
        # Flatten and single FC layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # Return logits
        
        return x
    
if __name__ == "__main__":
    model = PlantCNN(num_classes=39)
    
    dummy_input = torch.randn(4, 3, 256, 256)
    output = model(dummy_input)
    
    print(f"Model output shape: {output.shape}")  # Should be [4, 39]
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")