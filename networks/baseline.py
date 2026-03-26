import torch
import torch.nn as nn



class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        
        # Layer 1: Input 2 channels (1+1), output 32
        self.conv11 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu11 = nn.ReLU()
        self.conv21 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu21 = nn.ReLU()
        
        # Layer 2: Input 32, output 32
        self.conv12 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu12 = nn.ReLU()
        self.conv22 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu22 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img1, img2):
        # Process img1 and img2 separately
        x1 = self.relu11(self.conv11(img1))
        x2 = self.relu12(self.conv12(img2))
        x1 = self.relu21(self.conv21(x1))
        x2 = self.relu22(self.conv22(x2))
        
        
        
        
        x = x1 + x2
        
        x = self.relu3(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x
    
    
if __name__ == "__main__":
    # Test thử model
    
    model = BaselineModel()
    img1 = torch.randn(16, 1, 256, 256)
    img2 = torch.randn(16, 1, 256, 256)
    output = model(img1, img2)
    print(f"Output shape: {output.shape}") # Expected: [1, 1, 224, 224]


    # print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")