"""
Gaussian Splatting Decoder Training Script
=================================
"""

import argparse
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
import lpips
from fused_ssim import fused_ssim
from torchvision.utils import save_image
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gsplat.rendering import rasterization
from gsplat.utils import save_ply

from dataset import GaussianFaceDecoderDataset
from model import CondGaussianSplatting
from utils import set_random_seed, load_ply_to_splats

# Suppress warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class TrainingConfig:
    """Configuration for training Gaussian Splatting models."""
    
    # Data paths
    data_root: str = "/scratch/shared/nersemble-data/EXP-1-head-frame0_export"
    max_subjects: int = 417
    ply_file_path: str = "pretrained_weights/averaged_model.ply"
    save_path: str = "checkpoints"
    
    # Training parameters
    max_epochs: int = 400
    batch_size: int = 1
    num_workers: int = 4
    
    # Learning rates
    mlp_lr: float = 1e-4
    w_lr: float = 1e-4
    base_lr: float = 0.0
    embedding_lr: float = 0.0
    
    # Loss weights
    l1_weight: float = 0.6
    ssim_weight: float = 0.3
    lpips_weight: float = 0.1
    
    # Regularization
    scale_reg: float = 0.0
    pos_reg: float = 0.0
    
    # Rendering parameters
    near_plane: float = 0.01
    far_plane: float = 1e10
    sh_degree: int = 3
    camera_model: str = "pinhole"
    
    # Optimization
    ga_step: int = 1  # Gradient accumulation steps
    
    # Rasterization options
    packed: bool = False
    sparse_grad: bool = False
    antialiased: bool = False
    
    # Network choices
    lpips_net: str = "alex"
    
    # Logging and saving
    log_every: int=1000
    save_every: int = 1000
    eval_every: int = 1000
    save_images: bool = True
    save_ply_files: bool = False
    
    # Random seed
    seed: int = 42
    
    # Image dimensions (will be set from data)
    image_height: int = 802
    image_width: int = 550


class GaussianSplattingTrainer:
    """
    Trainer class for Gaussian Splatting models, aligned with inference script.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.cfg = config
        
        # Set random seed for reproducibility
        set_random_seed(config.seed)
        
        # Initialize models
        self._initialize_models()
        
        # Setup dataset and dataloader
        self._setup_data()
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Create output directories
        self._setup_directories()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        print(f"Trainer initialized with {len(self.trainset)} training samples")
    
    def _initialize_models(self):
        """Initialize encoder and decoder models."""
        print("Initializing models...")
        
        # Initialize decoder
        self.decoder = CondGaussianSplatting(
            ply_path=self.cfg.ply_file_path
        ).to(device)
        
        
        # Load base splats for reference
        self.base_splats = load_ply_to_splats(self.cfg.ply_file_path).to(device)
        print(f"Models initialized. Number of Gaussians: {len(self.base_splats['means'])}")
    
    def _setup_data(self):
        """Setup dataset and dataloader."""
        print("Setting up dataset...")
        # Initialize dataset
        self.trainset = GaussianFaceDecoderDataset(
            data_root=self.cfg.data_root,
            seed=self.cfg.seed,
            max_subjects=int(self.cfg.max_subjects)
        )
        
        # Initialize W vectors from dataset
        w_tensor = self.trainset.w_vectors.clone().to(device)
        self.w_vectors = nn.Parameter(w_tensor, requires_grad=True)
        self.w_ids_to_idx = self.trainset.w_ids_to_idx
        
        # Create dataloader
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
        
        print(f"Dataset loaded: {len(self.trainset)} samples")
    
    def _setup_optimizers(self):
        """Setup optimizers for training."""
        self.optimizers = {}
        self.schedulers = {}
        
        # Setup W vector optimizer
        self.w_optimizer = self.decoder.setup_w_vector_optimizer(
            self.w_vectors,
            w_vector_lr=self.cfg.w_lr
        )
        
        # Setup MLP optimizers
        mlp_optimizers = self.decoder.setup_optimizers(
            base_lr=self.cfg.base_lr,
            conditioning_lr=self.cfg.mlp_lr,
            w_vector_lr=0.0,  # Already handled above
            embedding_lr=self.cfg.embedding_lr
        )
        
        # Store optimizers
        self.optimizers = {
            'w_vectors': self.w_optimizer,
            **mlp_optimizers
        }
        
        # Setup schedulers if needed
        for name, optimizer in self.optimizers.items():
            if optimizer is not None:
                self.schedulers[name] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.8, patience=20, verbose=True
                )
    
    def _initialize_metrics(self):
        """Initialize evaluation metrics."""
        self.metrics = {
            'ssim': StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
            'psnr': PeakSignalNoiseRatio(data_range=1.0).to(device),
        }
        
        # Initialize LPIPS
        if self.cfg.lpips_net == "alex":
            self.metrics['lpips'] = LearnedPerceptualImagePatchSimilarity(
                net_type="alex",
                normalize=True
            ).to(device)
        elif self.cfg.lpips_net == "vgg":
            self.metrics['lpips'] = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg",
                normalize=False
            ).to(device)
        else:
            raise ValueError(f"Unknown LPIPS network: {self.cfg.lpips_net}")
        
        # LPIPS loss for training
        self.lpips_loss = lpips.LPIPS(net='alex').to(device)
    
    def _setup_directories(self):
        """Create output directories for saving results."""
        # Create experiment name
        exp_name = f"decoder_mlp{self.cfg.mlp_lr}_w{self.cfg.w_lr}_base{self.cfg.base_lr}"
        
        # Create directory paths
        self.dirs = {
            'base': Path(self.cfg.save_path) / exp_name,
            'images': Path(self.cfg.save_path) / exp_name / 'images',
            'ply': Path(self.cfg.save_path) / exp_name / 'ply',
            'checkpoints': Path(self.cfg.save_path) / exp_name / 'checkpoints',
        }
        
        # Create directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directories created under: {self.dirs['base']}")
    
    def rasterize_splats(
        self,
        splats: Dict[str, torch.Tensor],
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Rasterize 3D Gaussian splats to 2D images.
        Aligned with inference script implementation.
        """
        # Extract splat parameters
        means = splats["means"]
        quats = splats["quats"]
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)
        
        # Compute view matrix
        try:
            viewmatrix = torch.linalg.inv(camtoworlds)
        except torch.linalg.LinAlgError:
            print("Warning: Singular camera matrix, using identity")
            viewmatrix = torch.eye(4, device=camtoworlds.device).unsqueeze(0)
        
        # Rasterize
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmatrix,
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=False,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode="antialiased" if self.cfg.antialiased else "classic",
            distributed=False,
            camera_model=self.cfg.camera_model,
            sh_degree=kwargs.get('sh_degree', self.cfg.sh_degree),
            near_plane=kwargs.get('near_plane', self.cfg.near_plane),
            far_plane=kwargs.get('far_plane', self.cfg.far_plane),
            render_mode=kwargs.get('render_mode', 'RGB'),
        )
        
        # Composite with white background
        white_background = torch.ones_like(render_colors)
        render_colors = render_colors * render_alphas + white_background * (1 - render_alphas)
        
        return render_colors, render_alphas, info
    
    def compute_losses(
        self,
        renders: torch.Tensor,
        targets: torch.Tensor,
        raw_outputs: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        Aligned with inference script implementation.
        """
        losses = {}
        
        # L1 loss
        losses['l1'] = F.l1_loss(renders, targets)
        
        # SSIM loss
        losses['ssim'] = 1.0 - fused_ssim(
            renders.permute(0, 3, 1, 2),
            targets.permute(0, 3, 1, 2),
            padding="valid"
        )
        
        # LPIPS loss
        losses['lpips'] = self.lpips_loss(
            renders.permute(0, 3, 1, 2),
            targets.permute(0, 3, 1, 2)
        ).mean()
        
        # Combined loss
        losses['total'] = (
            losses['l1'] * self.cfg.l1_weight +
            losses['ssim'] * self.cfg.ssim_weight +
            losses['lpips'] * self.cfg.lpips_weight
        ) / self.cfg.ga_step
        
        # Add regularization if needed
        if raw_outputs is not None:
            if self.cfg.scale_reg > 0.0 and "raw_scales" in raw_outputs:
                scale_reg = torch.exp(raw_outputs["raw_scales"]).mean()
                losses['scale_reg'] = scale_reg * self.cfg.scale_reg
                losses['total'] += losses['scale_reg']
            
            if self.cfg.pos_reg > 0.0 and "raw_means" in raw_outputs:
                pos_reg = torch.abs(raw_outputs["raw_means"]).mean()
                losses['pos_reg'] = pos_reg * self.cfg.pos_reg
                losses['total'] += losses['pos_reg']
        
        return losses
    
    def train_step(
        self,
        data: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        """
        # Extract data
        camtoworlds = data["camtoworlds"].float().to(device)
        Ks = data["K"].float().to(device)
        gt_pixels = data["pixels"].to(device) / 255.0
        means_3d = data["means"].to(device)
        image_ids = data["id"]
        
        # Get image dimensions
        height, width = gt_pixels.shape[1:3]
        
        # Get W vectors for this batch
        w_indices = [self.w_ids_to_idx[i] for i in image_ids]
        w_vectors = self.w_vectors[w_indices]
        
        # Set models to training mode
        self.decoder.train()
        
        # Forward pass
        splats, raw_outputs = self.decoder(w_vectors, step)
        splats["means"] = means_3d.squeeze(0) + splats["means"]
        
        # Render
        renders, alphas, _ = self.rasterize_splats(
            splats=splats,
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
        )
        
        # Compute losses
        losses = self.compute_losses(renders, gt_pixels, raw_outputs)
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        if hasattr(self.decoder, 'conditioning_mlp'):
            torch.nn.utils.clip_grad_norm_(self.decoder.conditioning_mlp.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.w_vectors, max_norm=1.0)
        
        # Optimization step
        for optimizer in self.optimizers.values():
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()
        
        # Save sample images periodically
        if self.cfg.save_images and step % self.cfg.log_every == 0:
            self._save_sample_images(renders, gt_pixels, step)
        
        return {k: v.item() for k, v in losses.items()}
    
    def _save_sample_images(
        self,
        renders: torch.Tensor,
        targets: torch.Tensor,
        step: int
    ):
        """Save sample rendered and target images."""
        # Save rendered image
        render_path = self.dirs['images'] / f"render_step_{step:08d}.png"
        save_image(
            renders.squeeze(0).permute(2, 0, 1),
            str(render_path)
        )
        
        # Save target image
        target_path = self.dirs['images'] / f"target_step_{step:08d}.png"
        save_image(
            targets.squeeze(0).permute(2, 0, 1),
            str(target_path)
        )
    
    def evaluate(self, step: int) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        """
        self.decoder.eval()
        
        eval_metrics = {
            'ssim': 0.0,
            'psnr': 0.0,
            'lpips': 0.0,
        }
        
        num_samples = 0
        
        with torch.no_grad():
            for data in self.trainloader:
                # For simplicity, using training data for evaluation
                # In practice, you'd have a separate validation set
                
                camtoworlds = data["camtoworlds"].float().to(device)
                Ks = data["K"].float().to(device)
                gt_pixels = data["pixels"].to(device) / 255.0
                means_3d = data["means"].to(device)
                image_ids = data["id"]
                
                height, width = gt_pixels.shape[1:3]
                
                # Get W vectors
                w_indices = [self.w_ids_to_idx[i] for i in image_ids]
                w_vectors = self.w_vectors[w_indices]
                
                # Forward pass
                splats, _ = self.decoder(w_vectors, step)
                splats["means"] = means_3d.squeeze(0) + splats["means"]
                
                # Render
                renders, _, _ = self.rasterize_splats(
                    splats=splats,
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                )
                
                # Compute metrics
                eval_metrics['ssim'] += self.metrics['ssim'](
                    renders.permute(0, 3, 1, 2),
                    gt_pixels.permute(0, 3, 1, 2)
                ).item()
                
                eval_metrics['psnr'] += self.metrics['psnr'](
                    renders.permute(0, 3, 1, 2),
                    gt_pixels.permute(0, 3, 1, 2)
                ).item()
                
                eval_metrics['lpips'] += self.metrics['lpips'](
                    renders.permute(0, 3, 1, 2),
                    gt_pixels.permute(0, 3, 1, 2)
                ).item()
                
                num_samples += 1
                
                # Only evaluate on a subset for efficiency
                if num_samples >= 10:
                    break
        
        # Average metrics
        for key in eval_metrics:
            eval_metrics[key] /= num_samples
        
        return eval_metrics
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "model_dict": self.decoder.state_dict(),
            "w_vectors": self.w_vectors,
            "w_ids_to_idx": self.w_ids_to_idx,
            "epoch": epoch,
            "global_step": self.global_step,
            "config": self.cfg,
        }
        
        # Save latest checkpoint
        latest_path = self.dirs['checkpoints'] / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save epoch-specific checkpoint
        epoch_path = self.dirs['checkpoints'] / f"epoch_{epoch:04d}.pth"
        torch.save(checkpoint, epoch_path)
        
        print(f"Checkpoint saved: {epoch_path}")
    
    def run(self):
        """Main training loop."""
        print(f"\nStarting training for {self.cfg.max_epochs} epochs...")
        
        for epoch in range(self.cfg.max_epochs):
            epoch_start_time = time.time()
            epoch_losses = []
            
            # Progress bar for the epoch
            pbar = tqdm.tqdm(
                self.trainloader,
                desc=f"Epoch {epoch+1}/{self.cfg.max_epochs}"
            )
            
            for batch_idx, data in enumerate(pbar):
                # Training step
                losses = self.train_step(data, self.global_step)
                epoch_losses.append(losses['total'])
                
                # Update progress bar
                if losses:
                    loss_str = " ".join([
                        f"{k}: {v:.4f}" for k, v in losses.items()
                        if k != 'total'
                    ])
                    pbar.set_postfix_str(f"Loss: {losses['total']:.4f} | {loss_str}")
                
                self.global_step += 1
            
            # Compute average epoch loss
            avg_loss = np.mean(epoch_losses)
            epoch_time = time.time() - epoch_start_time
            
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")
            
            # Learning rate scheduling
            for scheduler in self.schedulers.values():
                scheduler.step(avg_loss)
            
            # Evaluation
            if (epoch + 1) % self.cfg.eval_every == 0:
                eval_metrics = self.evaluate(self.global_step)
                print(f"Eval - SSIM: {eval_metrics['ssim']:.4f}, "
                      f"PSNR: {eval_metrics['psnr']:.4f}, "
                      f"LPIPS: {eval_metrics['lpips']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint(epoch + 1)
            
            # Save PLY files if requested
            if self.cfg.save_ply_files and (epoch + 1) % self.cfg.save_every == 0:
                # Get a sample to save PLY
                with torch.no_grad():
                    data = next(iter(self.trainloader))
                    image_ids = data["id"]
                    w_indices = [self.w_ids_to_idx[i.item()] for i in image_ids]
                    w_vectors = self.w_vectors[w_indices]
                    
                    splats, _ = self.decoder(w_vectors, self.global_step)
                    ply_path = self.dirs['ply'] / f"epoch_{epoch+1:04d}.ply"
                    save_ply(splats, str(ply_path), None)
            
            self.current_epoch = epoch + 1
        
        # Final checkpoint
        self.save_checkpoint(self.current_epoch)
        print("\nTraining completed!")
        print(f"Results saved to: {self.dirs['base']}")


def create_config_from_args() -> TrainingConfig:
    """
    Create configuration from command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Gaussian Splatting Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        '--data_root', type=str, default="data",
        help='Path to the dataset root directory'
    )
    parser.add_argument(
        '--max_subjects', type=str, default="417",
        help='Max subjects for quicker training/validation'
    )
    parser.add_argument(
        '--ply_file_path', type=str,
        default="pretrained_weights/averaged_model.ply",
        help='Path to base PLY model file'
    )
    parser.add_argument(
        '--save_path', type=str, default="checkpoints",
        help='Directory for saving results'
    )
    
    # Training parameters
    parser.add_argument(
        '--max_epochs', type=int, default=400,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers'
    )
    
    # Learning rates
    parser.add_argument(
        '--mlp_lr', type=float, default=1e-4,
        help='Learning rate for MLP'
    )
    parser.add_argument(
        '--w_lr', type=float, default=1e-4,
        help='Learning rate for W vectors'
    )
    parser.add_argument(
        '--base_lr', type=float, default=0.0,
        help='Learning rate for base Gaussians'
    )
    parser.add_argument(
        '--embedding_lr', type=float, default=0.0,
        help='Learning rate for embeddings'
    )
    
    # Loss weights
    parser.add_argument(
        '--l1_weight', type=float, default=0.6,
        help='Weight for L1 loss'
    )
    parser.add_argument(
        '--ssim_weight', type=float, default=0.3,
        help='Weight for SSIM loss'
    )
    parser.add_argument(
        '--lpips_weight', type=float, default=0.1,
        help='Weight for LPIPS loss'
    )
    
    # Regularization
    parser.add_argument(
        '--scale_reg', type=float, default=0.0,
        help='Scale regularization weight'
    )
    parser.add_argument(
        '--pos_reg', type=float, default=0.0,
        help='Position regularization weight'
    )
    
    # Rendering parameters
    parser.add_argument(
        '--near_plane', type=float, default=0.01,
        help='Near plane clipping distance'
    )
    parser.add_argument(
        '--far_plane', type=float, default=1e10,
        help='Far plane clipping distance'
    )
    parser.add_argument(
        '--sh_degree', type=int, default=3,
        help='Spherical harmonics degree'
    )
    parser.add_argument(
        '--camera_model', type=str, default="pinhole",
        choices=["pinhole", "ortho", "fisheye"],
        help='Camera model type'
    )
    
    # Boolean flags
    parser.add_argument(
        '--packed', action='store_true',
        help='Use packed mode for rasterization'
    )
    parser.add_argument(
        '--sparse_grad', action='store_true',
        help='Use sparse gradients'
    )
    parser.add_argument(
        '--antialiased', action='store_true',
        help='Use antialiasing in rasterization'
    )
    parser.add_argument(
        '--save_ply_files', action='store_true',
        help='Save PLY files during training'
    )
    
    # Logging parameters
    parser.add_argument(
        '--log_every', type=int, default=1000,
        help='Log frequency (steps)'
    )
    parser.add_argument(
        '--save_every', type=int, default=1000,
        help='Save frequency (epochs)'
    )
    parser.add_argument(
        '--eval_every', type=int, default=1000,
        help='Evaluation frequency (epochs)'
    )
    
    # Network choices
    parser.add_argument(
        '--lpips_net', type=str, default="alex",
        choices=["alex", "vgg"],
        help='LPIPS network type'
    )
    
    # Random seed
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = TrainingConfig(**vars(args))
    
    return config


def main():
    """Main entry point for training."""
    # Parse arguments and create config
    config = create_config_from_args()
    
    # Print configuration
    print("\n" + "="*50)
    print("Training Configuration")
    print("="*50)
    for key, value in vars(config).items():
        print(f"{key:20s}: {value}")
    print("="*50 + "\n")
    
    # Create and run trainer
    trainer = GaussianSplattingTrainer(config)
    trainer.run()


if __name__ == "__main__":
    """
    Usage examples:
    
    # Basic usage with defaults
    python scripts/train.py
    
    # Custom learning rates
    python scripts/train.py --mlp_lr 2e-4 --w_lr 5e-5
    
    # With regularization
    python scripts/train.py --scale_reg 0.01 --pos_reg 0.001
    
    # Custom paths and epochs
    python scripts/train.py --data_root /path/to/data --max_epochs 800
    
    # Enable PLY saving and special modes
    python scripts/train.py --save_ply_files --packed --sparse_grad
    """
    main()