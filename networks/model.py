import torch
import torch.nn as nn

class SimpleSynthesisModel(nn.Module):
    def __init__(self):
        super(SimpleSynthesisModel, self).__init__()
        
        # Layer 1: Input 2 channels (1+1), output 32
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Input 32, output 32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Layer 3: 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()

        # Layer 6: Output 6 channels (RGB image), followed by Sigmoid
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img1, img2):
        # Concatenate theo chiều channel (dim=1 cho PyTorch NCHW)
        x = torch.cat((img1, img2), dim=1)
        
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.sigmoid(self.conv6(x))
        return x


class ModelDualChannel(nn.Module):
    def __init__(self):
        super(ModelDualChannel, self).__init__()
        
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
        
        
        
        # Concatenate theo chiều channel (dim=1 cho PyTorch NCHW)
        x = x1 + x2
        
        x = self.relu3(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x
    


class SimpleSynthesisModel10M(nn.Module):
    def __init__(self):
        super(SimpleSynthesisModel10M, self).__init__()
        
        # Layer 1: Input 2 channels (1+1), output 32
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        # Layer 2: Input 32, output 32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Layer 3: 
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.relu3 = nn.ReLU()

        # self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        # self.relu4 = nn.ReLU()

        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        # self.relu5 = nn.ReLU()

        # Layer 3: Output 3 channels (RGB image), followed by Sigmoid
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img1, img2):
        # Concatenate theo chiều channel (dim=1 cho PyTorch NCHW)
        x = torch.cat((img1, img2), dim=1)
        
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        # x = self.relu3(self.conv3(x))
        # x = self.relu4(self.conv4(x))
        # x = self.relu5(self.conv5(x))
        x = self.sigmoid(self.conv3(x))
        return x
    
if __name__ == "__main__":
    # Test thử model
    # model = SimpleSynthesisModel()
    model = ModelDualChannel()
    img1 = torch.randn(16, 1, 224, 224)
    img2 = torch.randn(16, 1, 224, 224)
    output = model(img1, img2)
    print(f"Output shape: {output.shape}") # Expected: [1, 1, 224, 224]


    # print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")