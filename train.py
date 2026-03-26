"""
Training script for Medical Image Fusion
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from losses.ECINloss import ssim_ir, ssim_vi,RMI_ir,RMI_vi,Hub_vi,Hub_ir
from losses.DATFuseloss import ir_loss, vi_loss,ssim_loss,gra_loss

# from losses.quangnamloss import ssim_loss_ir, ssim_loss_vi, sf_loss_ir, sf_loss_vi


# Import custom modules
from networks.model import SimpleSynthesisModel, SimpleSynthesisModel10M, ModelDualChannel
from networks.fullmodel import FullModel
from networks.baseline import BaselineModel 
from dataset import create_dataloader
from utils import calculate_ssim, save_checkpoint, load_checkpoint


def total_loss(input_ir, input_vi, fused_result, alpha = 1,lamda = 100,gamma = 10):
    _ir_loss = ir_loss(fused_result, input_ir)
    _vi_loss = vi_loss(fused_result, input_vi)
    _ssim_loss = ssim_loss(fused_result, input_ir, input_vi)
    _gra_loss = gra_loss(fused_result, input_ir, input_vi)
    total = _ir_loss + alpha * _vi_loss + gamma *_ssim_loss + lamda * _gra_loss
    return total, _ir_loss, _vi_loss, _ssim_loss, _gra_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train Medical Image Fusion Model')
    
    # Dataset arguments
    parser.add_argument('-m','--modality', type=str, default='CT-MRI',
                        choices=['CT-MRI', 'PET-MRI', 'SPECT-MRI'],
                        help='Type of modality to train')
    parser.add_argument('--data_root', type=str, 
                        default=r'.\datasets\DatasetBMP',
                        help='Path to dataset root directory')
    
    # Model arguments
    parser.add_argument('-n','--method_name', type=str, required=True,
                        help='Name of the method (used for saving checkpoints and logs)')
    
    # Training arguments
    parser.add_argument('-e','--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('-b','--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint_dir', type=str, default=r'.\checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default=r'.\logs',
                        help='Directory to save tensorboard logs')
    parser.add_argument('-s','--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')

    # parser.add_argument('--weight', default=[1.5, 1,0,0,0,0,1,1.5], type=float)
    # parser.add_argument('--weight', default=[1, 0.05, 0.0006, 0.00025], type=float)
    parser.add_argument('--weight', default=[1.5, 1,0,0,0,0,1,1.5], type=float)
    
    
    args = parser.parse_args()
    return args


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, args):
    """Train for one epoch."""
    model.train()
    
    epoch_loss = 0.0
    epoch_ssim = 0.0
    num_batches = len(dataloader)

    
    # ssim_ir, ssim_vi,RMI_ir,RMI_vi,Hub_ir,Hub_vi,mse_ir,mse_vi = criterion[0],criterion[1],criterion[2],criterion[3],criterion[4],criterion[5],criterion[6],criterion[7]
    # ssim_loss_ir, ssim_loss_vi, sf_loss_ir, sf_loss_vi = criterion[0],criterion[1],criterion[2],criterion[3]
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for batch_idx, batch_data in enumerate(pbar):
        # Handle different modalities
        if args.modality == 'CT-MRI':
            img1, img2 = batch_data  # A (CT), B (MRI)
        else:  # PET-MRI or SPECT-MRI
            img1, img2, cbcr = batch_data  # A (Y channel), B (MRI), C (CbCr)
        
        # Move to device
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(img1, img2)
        
        # Calculate loss (MSE between output and average of inputs as baseline)
        # For a simple baseline, we can use average of two inputs as pseudo ground truth
        # In practice, you might want to use a different strategy
        # target = (img1 + img2) / 2.0

        # loss1,loss2 = criterion(output, img1),criterion(output,img2)
        # loss = (loss1+loss2)*0.5
        
        # use loss function of ECINFusion

        loss, _ir_loss, _vi_loss, _ssim_loss, _gra_loss = total_loss(img1, img2, output, alpha = 1,lamda = 100,gamma = 10)
        # loss_ssim_ir= args.weight[0] * ssim_ir(output, img1)
        # loss_ssim_vi= args.weight[1] * ssim_vi(output, img2)
        # loss_RMI_ir= args.weight[2] * RMI_ir(output,img1)
        # loss_RMI_vi = args.weight[3] * RMI_vi(output,img2)
        # loss_Hub_ir = args.weight[4] * Hub_ir(output, img1)
        # loss_Hub_vi = args.weight[5] * Hub_vi(output, img2)
        # loss_mse_ir = args.weight[6] * mse_ir(output, img1)
        # loss_mse_vi = args.weight[7] * mse_vi(output, img2)
        # loss_ssim_ir= args.weight[0] * ssim_loss_ir(output, img1)
        # loss_ssim_vi= args.weight[1] * ssim_loss_vi(output, img2)
        # loss_sf_ir= args.weight[2] * sf_loss_ir(output, img1)
        # loss_sf_vi= args.weight[3] * sf_loss_vi(output, img2)
        
        
        # print(f"EPOCH: {epoch}/{args.epochs} - Batch: {batch_idx}/{num_batches} - Loss_SSIM_IR: {loss_ssim_ir.item():.4f} - Loss_SSIM_VI: {loss_ssim_vi.item():.4f} - Loss_RMI_IR: {loss_RMI_ir.item():.4f} - Loss_RMI_VI: {loss_RMI_vi.item():.4f} - Loss_Hub_IR: {loss_Hub_ir.item():.4f} - Loss_Hub_VI: {loss_Hub_vi.item():.4f}")
        
        # loss = loss_ssim_ir + loss_ssim_vi+loss_RMI_ir+ loss_RMI_vi+loss_Hub_vi+loss_Hub_ir + loss_mse_ir + loss_mse_vi
        # loss = loss_ssim_ir + loss_ssim_vi+loss_sf_ir+ loss_sf_vi

        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate SSIM
        with torch.no_grad():
            ssim_value1,ssim_value2 = calculate_ssim(output, img1),calculate_ssim(output, img2)
            ssim_value = 0.5*(ssim_value1+ssim_value2)
        
        # Update statistics
        epoch_loss += loss.item()
        epoch_ssim += ssim_value
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            # 'loss_ssim_ir': f'{loss_ssim_ir.item():.4f}',
            # 'loss_ssim_vi': f'{loss_ssim_vi.item():.4f}',
            # 'loss_sf_ir': f'{loss_sf_ir.item():.4f}',
            # 'loss_sf_vi': f'{loss_sf_vi.item():.4f}',
            # 'loss_RMI_ir': f'{loss_RMI_ir.item():.4f}',
            # 'loss_RMI_vi': f'{loss_RMI_vi.item():.4f}',
            # 'loss_Hub_ir': f'{loss_Hub_ir.item():.4f}',
            # 'loss_Hub_vi': f'{loss_Hub_vi.item():.4f}',
            # 'loss_mse_ir': f'{loss_mse_ir.item():.4f}',
            # 'loss_mse_vi': f'{loss_mse_vi.item():.4f}',
            'ir_loss': f'{_ir_loss.item():.4f}',
            'vi_loss': f'{_vi_loss.item():.4f}',
            'ssim_loss': f'{_ssim_loss.item():.4f}',
            'gra_loss': f'{_gra_loss.item():.4f}',
            'ssim': f'{ssim_value:.4f}'
        })
        
        # Log to tensorboard (per batch)
        global_step = epoch * num_batches + batch_idx
        writer.add_scalar('train/batch_loss', loss.item(), global_step)
        writer.add_scalar('train/batch_ssim', ssim_value, global_step)
        writer.add_scalar('train/ir_mse_loss', _ir_loss.item(), global_step)
        writer.add_scalar('train/vi_mse_loss', _vi_loss.item(), global_step)
        writer.add_scalar('train/ssim_loss', _ssim_loss.item(), global_step)
        writer.add_scalar('train/gra_loss', _gra_loss.item(), global_step)
        # writer.add_scalar('train/loss_ssim_ir', loss_ssim_ir.item(), global_step)
        # writer.add_scalar('train/loss_ssim_vi', loss_ssim_vi.item(), global_step)
        # writer.add_scalar('train/loss_RMI_ir', loss_RMI_ir.item(), global_step)
        # writer.add_scalar('train/loss_RMI_vi', loss_RMI_vi.item(), global_step)
        # writer.add_scalar('train/loss_Hub_ir', loss_Hub_ir.item(), global_step)
        # writer.add_scalar('train/loss_Hub_vi', loss_Hub_vi.item(), global_step)
        # writer.add_scalar('train/loss_mse_ir', loss_mse_ir.item(), global_step)
        # writer.add_scalar('train/loss_mse_vi', loss_mse_vi.item(), global_step)
        # writer.add_scalar('train/loss_sf_ir', loss_sf_ir.item(), global_step)
        # writer.add_scalar('train/loss_sf_vi', loss_sf_vi.item(), global_step)

    
    # Calculate epoch averages
    avg_loss = epoch_loss / num_batches
    avg_ssim = epoch_ssim / num_batches
    
    # Log epoch statistics
    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/ssim', avg_ssim, epoch)
    
    return avg_loss, avg_ssim


def validate(model, dataloader, criterion, device, args):
    """Validate the model."""
    model.eval()
    
    val_loss = 0.0
    val_ssim = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        # ssim_ir, ssim_vi,RMI_ir,RMI_vi,Hub_ir,Hub_vi,mse_ir,mse_vi = criterion[0],criterion[1],criterion[2],criterion[3],criterion[4],criterion[5],criterion[6],criterion[7]
        # ssim_ir, ssim_vi, sf_loss_ir, sf_loss_vi = criterion[0],criterion[1],criterion[2],criterion[3]
        for batch_data in tqdm(dataloader, desc='Validation'):
            # Handle different modalities
            if args.modality == 'CT-MRI':
                img1, img2 = batch_data
            else:
                img1, img2, cbcr = batch_data
            
            img1 = img1.to(device)
            img2 = img2.to(device)
            
            # Forward pass
            output = model(img1, img2)
            
            # Calculate metrics
            # target = (img1 + img2) / 2.0
            loss, _ir_loss, _vi_loss, _ssim_loss, _gra_loss = total_loss(img1, img2, output, alpha = 1,lamda = 100,gamma = 10)

            # loss1, loss2 = criterion(output, img1), criterion(output, img2)
            # loss_ssim_ir= args.weight[0] * ssim_ir(output, img1)
            # loss_ssim_vi= args.weight[1] * ssim_vi(output, img2)
            # loss_RMI_ir= args.weight[2] * RMI_ir(output,img1)
            # loss_RMI_vi = args.weight[3] * RMI_vi(output,img2)
            # loss_Hub_ir = args.weight[4] * Hub_ir(output, img1)
            # loss_Hub_vi = args.weight[5] * Hub_vi(output, img2)
            # loss_mse_ir = args.weight[6] * mse_ir(output, img1)
            # loss_mse_vi = args.weight[7] * mse_vi(output, img2)
            # loss_sf_ir= args.weight[2] * sf_loss_ir(output, img1)
            # loss_sf_vi= args.weight[3] * sf_loss_vi(output, img2)
            
            # loss = loss_ssim_ir + loss_ssim_vi+loss_RMI_ir+ loss_RMI_vi+loss_Hub_vi+loss_Hub_ir+ loss_mse_ir + loss_mse_vi
            # loss = loss_ssim_ir + loss_ssim_vi+loss_sf_ir+ loss_sf_vi
            
            # loss = (loss1 + loss2)*0.5
            ssim_value1,ssim_value2 = calculate_ssim(output, img1),calculate_ssim(output, img2)
            ssim_value = 0.5*(ssim_value1+ssim_value2)
            val_loss += loss.item()
            val_ssim += ssim_value
    
    avg_loss = val_loss / num_batches
    avg_ssim = val_ssim / num_batches
    
    return avg_loss, avg_ssim


def main():
    # Parse arguments
    args = parse_args()
    print(args)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories for checkpoints and logs
    checkpoint_path = os.path.join(args.checkpoint_dir, args.method_name)
    log_path = os.path.join(args.log_dir, args.method_name)
    
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    print(f"Checkpoints will be saved to: {checkpoint_path}")
    print(f"Logs will be saved to: {log_path}")
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=log_path)
    
    # Create dataloaders
    print(f"\nLoading {args.modality} dataset...")
    train_loader = create_dataloader(
        root_dir=args.data_root,
        modality=args.modality,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        transform=None
    )
    
    val_loader = create_dataloader(
        root_dir=args.data_root,
        modality=args.modality,
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        transform=None
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    # model = SimpleSynthesisModel10M()
    # model = ModelDualChannel()
    model = FullModel()
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: ModelDualChannel")
    print(f"Total parameters: {num_params:,}")
    
    # Loss function and optimizer
    criterion_MSE_ir, criterion_MSE_vi = nn.MSELoss(), nn.MSELoss()
    criterions_ECINFusion_MSE = [ssim_ir, ssim_vi,RMI_ir,RMI_vi,Hub_ir,Hub_vi,criterion_MSE_ir,criterion_MSE_vi]
    # criterions_quangnam = [ssim_loss_ir, ssim_loss_vi, sf_loss_ir, sf_loss_vi]
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_ssim = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        # train_loss, train_ssim = train_one_epoch(
        #     model, train_loader, criterion, optimizer, 
        #     device, epoch, writer, args
        # )

        train_loss, train_ssim = train_one_epoch(
            model, train_loader, None, optimizer, 
            device, epoch, writer, args
        )
        
        # Validate
        print("Validating...")
        val_loss, val_ssim = validate(model, val_loader, None, device, args)
        
        # Log validation metrics
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/ssim', val_ssim, epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, SSIM: {train_ssim:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, SSIM: {val_ssim:.4f}")
        print("-" * 60)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_file = os.path.join(checkpoint_path, f'epoch_{epoch+1:03d}.pth')
            save_checkpoint(checkpoint_file, model, optimizer, epoch+1)
            print(f"Saved checkpoint: {checkpoint_file}")
        
        # Save best model
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            best_checkpoint = os.path.join(checkpoint_path, 'best_model.pth')
            save_checkpoint(best_checkpoint, model, optimizer, epoch+1)
            print(f"Saved best model with SSIM: {best_ssim:.4f}")
        
        # Save latest checkpoint
        latest_checkpoint = os.path.join(checkpoint_path, 'latest.pth')
        save_checkpoint(latest_checkpoint, model, optimizer, epoch+1)
    
    # Training finished
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation SSIM: {best_ssim:.4f}")
    print("="*60)
    
    writer.close()


if __name__ == '__main__':
    main()
