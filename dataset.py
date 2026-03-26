"""
PyTorch Dataset for Medical Image Fusion
Supports CT-MRI, PET-MRI, and SPECT-MRI datasets
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class MedicalImageFusionDataset(Dataset):
    """
    Dataset for medical image fusion.
    
    For CT-MRI:
        - Returns (A, B) where A=CT grayscale, B=MRI grayscale
        
    For PET-MRI / SPECT-MRI:
        - Returns (A, B, C) where:
            A = Y channel of PET/SPECT (from YCbCr)
            B = MRI grayscale
            C = CbCr channels of PET/SPECT
    
    Args:
        root_dir: Path to dataset root (e.g., 'datasets/DatasetBMP')
        modality: One of 'CT-MRI', 'PET-MRI', 'SPECT-MRI'
        split: 'train' or 'test'
        transform: Optional transform to be applied on images
    """
    
    def __init__(self, root_dir, modality='CT-MRI', split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.modality = modality
        self.split = split
        self.transform = transform
        
        
        # Validate modality
        if modality not in ['CT-MRI', 'PET-MRI', 'SPECT-MRI']:
            raise ValueError(f"Invalid modality: {modality}. Must be one of ['CT-MRI', 'PET-MRI', 'SPECT-MRI']")
        
        # Validate split
        if split not in ['train', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
        
        # Determine modality type and paths
        if modality == 'CT-MRI':
            self.is_color_modality = False
            self.modality1_name = 'CT'
            self.modality2_name = 'MRI'
        elif modality == 'PET-MRI':
            self.is_color_modality = True
            self.modality1_name = 'PET'
            self.modality2_name = 'MRI'
        elif modality == 'SPECT-MRI':
            self.is_color_modality = True
            self.modality1_name = 'SPECT'
            self.modality2_name = 'MRI'
        
        # Set paths
        self.modality_dir = self.root_dir / modality / split
        self.path_modality1 = self.modality_dir / self.modality1_name
        self.path_modality2 = self.modality_dir / self.modality2_name
        
        # Validate paths
        if not self.path_modality1.exists():
            raise FileNotFoundError(f"Modality 1 path not found: {self.path_modality1}")
        if not self.path_modality2.exists():
            raise FileNotFoundError(f"Modality 2 path not found: {self.path_modality2}")
        
        # Get list of image files
        self.images_modality1 = sorted([f for f in os.listdir(self.path_modality1) if f.endswith('.bmp')])
        self.images_modality2 = sorted([f for f in os.listdir(self.path_modality2) if f.endswith('.bmp')])
        
        # Ensure both modalities have the same number of images
        if len(self.images_modality1) != len(self.images_modality2):
            raise ValueError(f"Number of images mismatch: {len(self.images_modality1)} vs {len(self.images_modality2)}")
        
        # Ensure filenames match
        for img1, img2 in zip(self.images_modality1, self.images_modality2):
            if img1 != img2:
                raise ValueError(f"Filename mismatch: {img1} vs {img2}")
        
        print(f"Loaded {modality} {split} dataset: {len(self.images_modality1)} image pairs")
    
    def __len__(self):
        return len(self.images_modality1)
    
    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Returns:
            For CT-MRI: (A, B) tuple where A=CT, B=MRI
            For PET-MRI/SPECT-MRI: (A, B, C) tuple where A=Y channel, B=MRI, C=CbCr channels
        """
        # Get image filenames
        img1_name = self.images_modality1[idx]
        img2_name = self.images_modality2[idx]
        
        img1_path = os.path.join(self.path_modality1, img1_name)
        img2_path = os.path.join(self.path_modality2, img2_name)
        
        if self.is_color_modality:
            # For PET-MRI / SPECT-MRI: read color image and convert to YCbCr
            img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)  # BGR
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
            
            if img1 is None:
                raise FileNotFoundError(f"Failed to load image: {img1_path}")
            if img2 is None:
                raise FileNotFoundError(f"Failed to load image: {img2_path}")
            
            # Convert BGR to RGB, then to YCbCr
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1_ycbcr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2YCrCb)
            
            # Split into Y and CbCr channels
            Y = img1_ycbcr[:, :, 0]  # Y channel (uint8)
            CbCr = img1_ycbcr[:, :, 1:]  # CbCr channels (uint8, H, W, 2)
            
            # Convert Y and MRI to float32 and normalize to [0, 1] for training
            A = Y.astype(np.float32) / 255.0  # Y channel
            B = img2.astype(np.float32) / 255.0  # MRI
            
            # Keep CbCr as uint8 [0, 255] for easy reconstruction during inference
            C = CbCr  # uint8, range [0, 255], shape (H, W, 2)
            
            # Add channel dimension for Y and MRI (H, W) -> (1, H, W)
            A = np.expand_dims(A, axis=0)
            B = np.expand_dims(B, axis=0)
            
            # Convert to PyTorch tensors
            A = torch.from_numpy(A)
            B = torch.from_numpy(B)
            
            # Apply transforms if provided
            if self.transform:
                A = self.transform(A)
                B = self.transform(B)
            
            return A, B, C
            
        else:
            # For CT-MRI: read both as grayscale
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # CT
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)  # MRI
            
            if img1 is None:
                raise FileNotFoundError(f"Failed to load image: {img1_path}")
            if img2 is None:
                raise FileNotFoundError(f"Failed to load image: {img2_path}")
            
            # Convert to float and normalize to [0, 1]
            A = img1.astype(np.float32) / 255.0
            B = img2.astype(np.float32) / 255.0
            
            # Add channel dimension (H, W) -> (1, H, W)
            A = np.expand_dims(A, axis=0)
            B = np.expand_dims(B, axis=0)
            
            # Convert to PyTorch tensors
            A = torch.from_numpy(A)
            B = torch.from_numpy(B)
            
            # Apply transforms if provided
            if self.transform:
                A = self.transform(A)
                B = self.transform(B)
            
            return A, B
    
    def get_image_name(self, idx):
        """Get the filename of an image pair."""
        return self.images_modality1[idx]


def create_dataloader(root_dir, modality='CT-MRI', split='train', batch_size=4, 
                     shuffle=True, num_workers=2, transform=None):
    """
    Create a DataLoader for medical image fusion.
    
    Args:
        root_dir: Path to dataset root
        modality: One of 'CT-MRI', 'PET-MRI', 'SPECT-MRI'
        split: 'train' or 'test'
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        transform: Optional transform
    
    Returns:
        DataLoader instance
    """
    dataset = MedicalImageFusionDataset(
        root_dir=root_dir,
        modality=modality,
        split=split,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def test_dataset():
    """Test the dataset implementation."""
    import matplotlib.pyplot as plt
    
    root_dir = r"C:\Users\Admin\MIF\MIF\datasets\DatasetBMP"
    
    print("="*60)
    print("Testing CT-MRI Dataset")
    print("="*60)
    
    # Test CT-MRI
    try:
        dataset_ct = MedicalImageFusionDataset(root_dir, modality='CT-MRI', split='train')
        print(f"CT-MRI dataset size: {len(dataset_ct)}")
        
        # Get first sample
        sample = dataset_ct[0]
        if len(sample) == 2:
            A, B = sample
            print(f"CT shape: {A.shape}, dtype: {A.dtype}, range: [{A.min():.3f}, {A.max():.3f}]")
            print(f"MRI shape: {B.shape}, dtype: {B.dtype}, range: [{B.min():.3f}, {B.max():.3f}]")
        
        # Test dataloader
        dataloader = create_dataloader(root_dir, modality='CT-MRI', split='train', 
                                      batch_size=8, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        if len(batch) == 2:
            A_batch, B_batch = batch
            print(f"Batch CT shape: {A_batch.shape}")
            print(f"Batch MRI shape: {B_batch.shape}")
    except Exception as e:
        print(f"Error testing CT-MRI: {e}")
    
    print("\n" + "="*60)
    print("Testing PET-MRI Dataset")
    print("="*60)
    
    # Test PET-MRI
    try:
        dataset_pet = MedicalImageFusionDataset(root_dir, modality='PET-MRI', split='train')
        print(f"PET-MRI dataset size: {len(dataset_pet)}")
        
        # Get first sample
        sample = dataset_pet[0]
        if len(sample) == 3:
            A, B, C = sample
            print(f"PET Y channel shape: {A.shape}, dtype: {A.dtype}, range: [{A.min():.3f}, {A.max():.3f}]")
            print(f"MRI shape: {B.shape}, dtype: {B.dtype}, range: [{B.min():.3f}, {B.max():.3f}]")
            print(f"PET CbCr channels shape: {C.shape}, dtype: {C.dtype}, range: [{C.min():.3f}, {C.max():.3f}]")
        
        # Test dataloader
        dataloader = create_dataloader(root_dir, modality='PET-MRI', split='train', 
                                      batch_size=8, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        if len(batch) == 3:
            A_batch, B_batch, C_batch = batch
            print(f"Batch PET Y shape: {A_batch.shape}")
            print(f"Batch MRI shape: {B_batch.shape}")
            print(f"Batch PET CbCr shape: {C_batch.shape}")
    except Exception as e:
        print(f"Error testing PET-MRI: {e}")
    
    print("\n" + "="*60)
    print("Testing SPECT-MRI Dataset")
    print("="*60)
    
    # Test SPECT-MRI
    try:
        dataset_spect = MedicalImageFusionDataset(root_dir, modality='SPECT-MRI', split='train')
        print(f"SPECT-MRI dataset size: {len(dataset_spect)}")
        
        # Get first sample
        sample = dataset_spect[0]
        if len(sample) == 3:
            A, B, C = sample
            print(f"SPECT Y channel shape: {A.shape}, dtype: {A.dtype}, range: [{A.min():.3f}, {A.max():.3f}]")
            print(f"MRI shape: {B.shape}, dtype: {B.dtype}, range: [{B.min():.3f}, {B.max():.3f}]")
            print(f"SPECT CbCr channels shape: {C.shape}, dtype: {C.dtype}, range: [{C.min():.3f}, {C.max():.3f}]")
    except Exception as e:
        print(f"Error testing SPECT-MRI: {e}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == '__main__':
    test_dataset()
