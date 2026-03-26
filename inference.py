"""
Inference script for Medical Image Fusion
"""

import os
import argparse
from pathlib import Path
import time

import torch
import numpy as np
import cv2
from tqdm import tqdm

# Import custom modules
from networks.model import SimpleSynthesisModel, SimpleSynthesisModel10M, ModelDualChannel
from networks.fullmodel import FullModel
from dataset import MedicalImageFusionDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for Medical Image Fusion Model')
    
    # Dataset arguments
    parser.add_argument('-d','--dataset_path', type=str, 
                        default=r'C:\Users\Admin\MIF\MIF\datasets\DatasetBMP',
                        help='Path to dataset root directory')
    
    # Model arguments
    parser.add_argument('-n','--method_name', type=str, required=True,
                        help='Name of the method (used for creating save directory)')
    parser.add_argument('-c','--checkpoint_dir', type=str, required=True,
                        help='Path to checkpoint file to load for inference')
    
    # Save arguments
    parser.add_argument('--save_path', type=str, 
                        default=r'C:\Users\Admin\MIF\MIF\results',
                        help='Base directory to save results')
    
    # Modality
    parser.add_argument('-m', '--modality', type=str, default=None,
                        choices=['CT-MRI', 'PET-MRI', 'SPECT-MRI', None],
                        help='Modality to run inference on. If None, runs on all modalities.')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    
    args = parser.parse_args()
    return args


def ycbcr_to_rgb(y_channel, cbcr_channels):
    """
    Convert Y channel and CbCr channels back to RGB image.
    
    Args:
        y_channel: numpy array (H, W) with values in [0, 255], dtype uint8
        cbcr_channels: numpy array (H, W, 2) with values in [0, 255], dtype uint8
    
    Returns:
        rgb_image: numpy array (H, W, 3) with values in [0, 255], dtype uint8
    """
    # Ensure y_channel is 2D
    if len(y_channel.shape) == 3:
        y_channel = y_channel.squeeze()
    
    # Stack Y and CbCr channels to create YCbCr image
    ycbcr = np.stack([y_channel, cbcr_channels[:, :, 0], cbcr_channels[:, :, 1]], axis=2)
    ycbcr = ycbcr.astype(np.uint8)
    
    # Convert YCbCr to RGB
    rgb = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)
    
    return rgb


def inference_single_modality(model, dataset, save_dir_gray, save_dir_rgb, device, modality_name):
    """
    Run inference on a single modality.
    
    Args:
        model: The trained model
        dataset: Dataset instance
        save_dir_gray: Directory to save grayscale fused images
        save_dir_rgb: Directory to save RGB fused images (only for PET-MRI and SPECT-MRI)
        device: Device to run inference on
        modality_name: Name of the modality (CT-MRI, PET-MRI, or SPECT-MRI)
    """
    model.eval()
    
    print(f"\nRunning inference on {modality_name}...")
    print(f"Number of images: {len(dataset)}")
    
    inference_times = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Processing {modality_name}"):
            # Get data
            batch_data = dataset[idx]
            
            if modality_name == 'CT-MRI':
                img1, img2 = batch_data  # A (CT), B (MRI)
                cbcr = None
            else:  # PET-MRI or SPECT-MRI
                img1, img2, cbcr = batch_data  # A (Y channel), B (MRI), C (CbCr)
            
            # Add batch dimension and move to device
            img1 = img1.unsqueeze(0).to(device)  # (1, 1, H, W)
            img2 = img2.unsqueeze(0).to(device)  # (1, 1, H, W)
            
            # Forward pass with timing
            start_time = time.time()
            output = model(img1, img2)
            torch.cuda.synchronize() if device == 'cuda' else None
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Convert output to numpy
            output_np = output.squeeze().cpu().numpy()  # (H, W)
            output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
            
            # Get image name
            img_name = dataset.get_image_name(idx)
            
            # Save grayscale image
            gray_save_path = os.path.join(save_dir_gray, img_name)
            cv2.imwrite(gray_save_path, output_np)
            
            # For PET-MRI and SPECT-MRI, also save RGB image
            if modality_name in ['PET-MRI', 'SPECT-MRI'] and cbcr is not None:
                # Reconstruct RGB image using fused Y channel and original CbCr
                rgb_image = ycbcr_to_rgb(output_np, cbcr)
                
                # Convert RGB to BGR for saving with OpenCV
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                # Save RGB image
                rgb_save_path = os.path.join(save_dir_rgb, img_name)
                cv2.imwrite(rgb_save_path, bgr_image)
    
    # Print inference statistics
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    print(f"\n{modality_name} Inference Statistics:")
    print(f"  Average inference time: {avg_time:.4f}s")
    print(f"  Std inference time: {std_time:.4f}s")
    print(f"  Total images processed: {len(dataset)}")
    print(f"  Results saved to:")
    print(f"    Grayscale: {save_dir_gray}")
    if modality_name in ['PET-MRI', 'SPECT-MRI']:
        print(f"    RGB: {save_dir_rgb}")
    
    return inference_times


def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint_dir}...")
    # model = SimpleSynthesisModel().to(device)
    # model = SimpleSynthesisModel10M().to(device)
    # model = ModelDualChannel().to(device)
    model = FullModel().to(device)

    checkpoint = torch.load(args.checkpoint_dir, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (no epoch info)")
    
    model.eval()
    print("Model loaded successfully!")
    
    # Determine which modalities to run inference on
    if args.modality is not None:
        modalities = [args.modality]
    else:
        modalities = ['CT-MRI', 'PET-MRI', 'SPECT-MRI']
    
    # Run inference for each modality
    for modality in modalities:
        print(f"\n{'='*60}")
        print(f"Processing {modality}")
        print(f"{'='*60}")
        
        # Create save directories
        base_save_dir = os.path.join(args.save_path, args.method_name)
        save_dir_gray = os.path.join(base_save_dir, "result_gray", modality)
        save_dir_rgb = os.path.join(base_save_dir, "result_rgb", modality)
        
        os.makedirs(save_dir_gray, exist_ok=True)
        if modality in ['PET-MRI', 'SPECT-MRI']:
            os.makedirs(save_dir_rgb, exist_ok=True)
        
        # Create dataset
        try:
            dataset = MedicalImageFusionDataset(
                root_dir=args.dataset_path,
                modality=modality,
                split='test',  # Use test split for inference
                transform=None
            )
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping {modality}...")
            continue
        
        # Run inference
        inference_times = inference_single_modality(
            model=model,
            dataset=dataset,
            save_dir_gray=save_dir_gray,
            save_dir_rgb=save_dir_rgb,
            device=device,
            modality_name=modality
        )
        
        # Save inference times to CSV
        csv_save_path = os.path.join(base_save_dir, f"inference_time_{args.method_name}_{modality}.csv")
        import pandas as pd
        df = pd.DataFrame({
            'image_idx': list(range(len(inference_times))),
            'inference_time_seconds': inference_times
        })
        df.to_csv(csv_save_path, index=False)
        print(f"Inference times saved to: {csv_save_path}")
    
    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
