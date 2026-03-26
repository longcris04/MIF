import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

class FoundationBranch(nn.Module):
    def __init__(self, model_type="vit_b", checkpoint="medsam_vit_b.pth"):
        super(FoundationBranch, self).__init__()
        # Load MedSAM và giữ lại Image Encoder
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.image_encoder = sam_model.image_encoder
        
        # Đóng băng trọng số nếu bạn không muốn fine-tune MedSAM để tiết kiệm VRAM
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # Nhánh Aggregation (theo sơ đồ): Gộp đặc trưng MRI và PET
        # MedSAM ViT-B thường output 256 channels
        self.aggregation_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), # 256*2 vì concat hoặc cộng
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=1) # Nén về 64 channels để sẵn sàng concat cuối cùng
        )

    def forward(self, mri, pet):
        # 1. Foundation Features (MedSAM yêu cầu input 1024x1024, bạn có thể cần resize trước)
        # Nếu input của bạn nhỏ hơn, MedSAM encoder vẫn chạy nhưng output size sẽ tỉ lệ thuận
        feat_mri = self.image_encoder(mri) # Output: [B, 256, H/16, W/16]
        feat_pet = self.image_encoder(pet)
        # print(f"feat_mri shape: {feat_mri.shape}") # Debug: kiểm tra kích thước đặc trưng sau encoder
        # exit(0)

        
        # 2. Aggregation (Gộp đặc trưng)
        combined_feat = torch.cat([feat_mri, feat_pet], dim=1)
        aggregated_feat = self.aggregation_conv(combined_feat)
        # print(f"aggregated_feat shape: {aggregated_feat.shape}") # Debug: kiểm tra kích thước sau aggregation
        
        # 3. Upsample để về cùng size với Baseline (H, W ban đầu)
        # Vì Encoder của SAM làm giảm resolution đi 16 lần
        # target_size = (mri.size(2), mri.size(3))
        target_size = (256,256)
        upsampled_feat = F.interpolate(aggregated_feat, size=target_size, mode='bilinear', align_corners=False)
        
        return upsampled_feat
    

if __name__ == "__main__":
    # Test thử branch Foundation
    w = 1024
    foundation_branch = FoundationBranch(checkpoint="medsam_vit_b.pth")
    mri = torch.randn(1, 3, w, w) # MedSAM yêu cầu input 3 channels RGB, bạn có thể duplicate kênh nếu chỉ có 1 channel
    pet = torch.randn(1, 3, w, w)
    output = foundation_branch(mri, pet)
    print(f"Output shape from Foundation Branch: {output.shape}") # Expected: [1, 64, 1024, 1024]