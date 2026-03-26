import torch
import torch.nn as nn
from networks.baseline import BaselineModel
from networks.MedSAM import FoundationBranch

class FullModel(nn.Module):
    def __init__(self):
        super(FullModel, self).__init__()
        self.foundation_branch = FoundationBranch(checkpoint=r"C:\Users\Admin\MIF\MIF\networks\medsam_vit_b.pth")
        self.baseline_branch = BaselineModel()
        # Layer tổng hợp cuối cùng: Input 64 (Foundation) + 1 (Baseline), output 1
        self.conv_final = nn.Conv2d(in_channels=65, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mri, pet):
        mri_medsam_input = torch.cat([mri, mri, mri], dim=1)
        pet_medsam_input = torch.cat([pet, pet, pet], dim=1)
        mri_medsam_input = nn.functional.interpolate(mri_medsam_input, size=(1024, 1024), mode='bilinear')
        pet_medsam_input = nn.functional.interpolate(pet_medsam_input, size=(1024, 1024), mode='bilinear') 
        # print(f"mri medsam shape: {mri_medsam_input.shape}") # Debug: kiểm tra kích thước input cho Foundation Branch
        # print(f"pet medsam shape: {pet_medsam_input.shape}") # Debug: kiểm tra kích thước input cho Foundation Branch

        foundation_feat = self.foundation_branch(mri_medsam_input, pet_medsam_input) # Output: [B, 64, H, W]
        baseline_output = self.baseline_branch(mri, pet) # Output: [B, 1, H, W]
        # print(f"Foundation Branch output shape: {foundation_feat.shape}") # Debug: kiểm tra kích thước output từ Foundation Branch
        # print(f"Baseline Branch output shape: {baseline_output.shape}") # Debug: kiểm tra
        # exit()
        # Concatenate theo chiều channel
        combined_feat = torch.cat([foundation_feat, baseline_output], dim=1) # Output: [B, 65, H, W]
        
        output = self.sigmoid(self.conv_final(combined_feat)) # Output: [B, 1, H, W]
        return output
    
if __name__ == "__main__":
    # Test thử model
    
    model = FullModel()
    mri = torch.randn(1, 1, 256, 256)
    pet = torch.randn(1, 1, 256, 256)
    output = model(mri, pet)
    print(f"Output shape: {output.shape}") # Expected: [16, 1, 256, 256]

    # print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")