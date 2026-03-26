"""
Utility functions for training and evaluation
"""

import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d
import numpy as np


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
        window_size: Size of the Gaussian window
        size_average: If True, return average SSIM across batch
    
    Returns:
        SSIM value (float)
    """
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    # Calculate means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def create_window(window_size, channel):
    """Create a Gaussian window for SSIM calculation."""
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                              for x in range(window_size)])
        return gauss / gauss.sum()
    
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def save_checkpoint(filepath, model, optimizer, epoch):
    """
    Save model checkpoint.
    
    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        epoch: Epoch number from checkpoint
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    return epoch


def denormalize(tensor, mean=0.0, std=1.0):
    """
    Denormalize a tensor.
    
    Args:
        tensor: Input tensor [B, C, H, W]
        mean: Mean value used for normalization
        std: Std value used for normalization
    
    Returns:
        Denormalized tensor
    """
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean).view(-1, 1, 1)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std).view(-1, 1, 1)
    
    return tensor * std + mean


def tensor_to_numpy(tensor):
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor [B, C, H, W] or [C, H, W]
    
    Returns:
        Numpy array [H, W, C] or [B, H, W, C]
    """
    if tensor.dim() == 4:
        # Batch of images [B, C, H, W] -> [B, H, W, C]
        return tensor.permute(0, 2, 3, 1).cpu().numpy()
    elif tensor.dim() == 3:
        # Single image [C, H, W] -> [H, W, C]
        return tensor.permute(1, 2, 0).cpu().numpy()
    else:
        return tensor.cpu().numpy()


def numpy_to_tensor(array):
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        array: Numpy array [H, W, C] or [B, H, W, C]
    
    Returns:
        PyTorch tensor [C, H, W] or [B, C, H, W]
    """
    if array.ndim == 4:
        # Batch [B, H, W, C] -> [B, C, H, W]
        return torch.from_numpy(array).permute(0, 3, 1, 2)
    elif array.ndim == 3:
        # Single image [H, W, C] -> [C, H, W]
        return torch.from_numpy(array).permute(2, 0, 1)
    else:
        return torch.from_numpy(array)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # Test SSIM calculation
    print("Testing SSIM calculation...")
    img1 = torch.rand(2, 1, 224, 224)
    img2 = torch.rand(2, 1, 224, 224)
    
    ssim_value = calculate_ssim(img1, img2)
    print(f"SSIM between random images: {ssim_value:.4f}")
    
    ssim_same = calculate_ssim(img1, img1)
    print(f"SSIM of image with itself: {ssim_same:.4f}")
