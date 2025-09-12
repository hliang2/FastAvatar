"""
Gaussian Splatting Training and Inference Module
================================================
This module provides training and inference capabilities for conditional
Gaussian Splatting models with face reconstruction.
"""

import argparse
import os
import time
import pdb
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

from dataset import GaussianFaceDataset
from model import CondGaussianSplatting, ViewInvariantEncoder
from utils import set_random_seed, load_ply_to_splats


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class TrainingConfig:
    """Configuration for training Gaussian Splatting models."""
    
    # Data paths
    data_root: str = "data"
    ply_file_path: str = "pretrained_weights/averaged_model.ply"
    save_path: str = "results"
    encoder_load_path: str = "pretrained_weights/encoder_neutral_flame.pth"
    decoder_load_path: str = "pretrained_weights/decoder_neutral_flame.pth"
    
    # Training parameters
    max_epochs: int = 801
    sample_id: int = 306
    batch_size: int = 1
    num_workers: int = 1
    
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
    camera_model: str = "pinhole"  # "pinhole", "ortho", or "fisheye"
    
    # Optimization
    ga_step: int = 1  # Gradient accumulation steps
    
    # Rasterization options
    packed: bool = False
    sparse_grad: bool = False
    antialiased: bool = False
    
    # Network choices
    lpips_net: str = "alex"  # "alex" or "vgg"
    
    # Random seed
    seed: int = 42
    
    # Image dimensions (will be set from data)
    image_height: int = 802
    image_width: int = 550


class GaussianSplattingTrainer:
    """
    Trainer class for Gaussian Splatting models.
    
    Handles model initialization, training loops, evaluation, and checkpointing.
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
        
        # Store all camera parameters from training set
        self.current_training_view_idx = None
        
    def _initialize_models(self):
        """Initialize encoder and decoder models."""
        print("Initializing models...")
        
        # Initialize encoder and decoder
        self.encoder = ViewInvariantEncoder().to(device)
        self.decoder = CondGaussianSplatting(
            ply_path=self.cfg.ply_file_path
        ).to(device)
        
        # Load pretrained weights if available
        if self.cfg.encoder_load_path and self.cfg.decoder_load_path:
            print("Loading pretrained models...")
            self._load_pretrained_models()
        
        # Load base splats for debugging
        self.base_splats = load_ply_to_splats(self.cfg.ply_file_path).to(device)
        print(f"Models initialized. Number of Gaussians: {len(self.base_splats['means'])}")
    
    def _load_pretrained_models(self):
        """Load pretrained encoder and decoder models."""
        try:
            # Load decoder
            decoder_checkpoint = torch.load(
                self.cfg.decoder_load_path,
                map_location=device
            )
            self.decoder.load_state_dict(decoder_checkpoint["model_dict"])
            
            # Load W vectors if available
            if "w_vectors" in decoder_checkpoint:
                self.w_vectors = decoder_checkpoint["w_vectors"].to(device)
                self.w_ids_to_idx = decoder_checkpoint.get("w_ids_to_idx", {})
            
            print(f"Decoder loaded from {self.cfg.decoder_load_path}")
            
            # Load encoder
            encoder_checkpoint = torch.load(
                self.cfg.encoder_load_path,
                map_location=device
            )
            self.encoder.load_state_dict(encoder_checkpoint["model_dict"])
            print(f"Encoder loaded from {self.cfg.encoder_load_path}")
            
        except FileNotFoundError as e:
            print(f"Warning: Could not load pretrained models: {e}")
            print("Continuing with random initialization...")
    
    def _setup_data(self):
        """Setup dataset and dataloader."""
        print("Setting up dataset...")
        
        # Dataset path
        data_root = os.path.join(self.cfg.data_root, str(self.cfg.sample_id))
        
        # Initialize dataset
        self.trainset = GaussianFaceDataset(
            data_root=data_root,
            seed=self.cfg.seed
        )
        
        # Initialize W vectors from dataset
        w_tensor = self.trainset.w_vectors.clone().to(device)
        self.w_vectors = nn.Parameter(w_tensor, requires_grad=True)
        self.w_ids_to_idx = self.trainset.w_ids_to_idx
        
        # Create dataloader
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
        
        # Collect all camera parameters from the dataset
        print("Collecting all camera parameters...")
        self.all_cameras = []
        for idx, data in enumerate(self.trainloader):
            camera_info = {
                'camtoworlds': data["camtoworlds"].float().to(device),
                'K': data["K"].float().to(device),
                'means_3d': data["means"].to(device),
                'view_idx': idx + 1
            }
            self.all_cameras.append(camera_info)
        
        print(f"Dataset loaded: {len(self.trainset)} samples")
        print(f"Collected {len(self.all_cameras)} camera viewpoints")
    
    def _setup_optimizers(self):
        """Setup optimizers for training."""
        self.optimizers = {}
        self.schedulers = {}
        
        # These will be initialized during training
        self.w_optimizer = None
        self.mlp_optimizer = None
    
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
        self.output_name = f"sample_{self.cfg.sample_id}"
        
        # Create directory paths
        self.dirs = {
            'images': Path(self.cfg.save_path) / 'images' / self.output_name,
            'ply': Path(self.cfg.save_path) / 'ply' / self.output_name,
        }
        
        # Create directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directories created under: {self.cfg.save_path}")
    
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
        
        Args:
            splats: Dictionary containing splat parameters
            camtoworlds: Camera-to-world transformation matrices
            Ks: Camera intrinsic matrices
            width: Image width
            height: Image height
            **kwargs: Additional rasterization parameters
            
        Returns:
            Tuple of (rendered colors, alpha values, info dict)
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
            rasterize_mode="classic",
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
        
        # Apply masks if provided
        masks = kwargs.get('masks')
        if masks is not None:
            render_colors[~masks] = 0
        
        return render_colors, render_alphas, info
    
    def render_all_views(
        self,
        splats: Dict[str, torch.Tensor],
        prefix: str,
        training_view: int,
        step: int
    ):
        """
        Render splats from all camera viewpoints and save images.
        
        Args:
            splats: Dictionary containing splat parameters
            prefix: Prefix for saved files (e.g., "initial" or "final")
            training_view: The view index that was used for training
            step: Current training step
        """
        print(f"Rendering {prefix} images from all {len(self.all_cameras)} viewpoints...")
        
        for camera_info in self.all_cameras:
            view_idx = camera_info['view_idx']
            camtoworlds = camera_info['camtoworlds']
            Ks = camera_info['K']
            
            # Render from this viewpoint
            renders, alphas, _ = self.rasterize_splats(
                splats=splats,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=self.cfg.image_width,
                height=self.cfg.image_height,
            )
            
            # Mark if this was the training view
            view_suffix = f"trainview{training_view:02d}_renderviewr{view_idx:02d}"
            if view_idx == training_view:
                view_suffix += "_TRAINED"
            
            # Save rendered image
            image_path = self.dirs['images'] / f"{prefix}_{view_suffix}_step{step:06d}.png"
            save_image(
                renders.squeeze(0).permute(2, 0, 1),
                str(image_path)
            )
        
        # Save PLY file once (same for all views)
        ply_path = self.dirs['ply'] / f"{prefix}_trainview{training_view:02d}_step{step:06d}.ply"
        save_ply(splats, str(ply_path), None)

    def compute_losses(
        self,
        renders: torch.Tensor,
        targets: torch.Tensor,
        raw_outputs: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            renders: Rendered images
            targets: Ground truth images
            raw_outputs: Raw network outputs for regularization
            
        Returns:
            Dictionary of loss values
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
            if self.cfg.scale_reg > 0.0 and "scales" in raw_outputs:
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
        epoch: int,
        view_idx: int
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            data: Batch data dictionary
            epoch: Current epoch
            view_idx: Current view index
            
        Returns:
            Dictionary of loss values
        """
        # Extract data
        camtoworlds = data["camtoworlds"].float().to(device)
        Ks = data["K"].float().to(device)
        gt_pixels = data["pixels"].to(device) / 255.0
        embedding = data["embedding"].to(device)
        means_3d = data["means"].to(device)
        
        # Store current training view
        self.current_training_view_idx = view_idx
        
        # Get image dimensions
        height, width = self.cfg.image_height, self.cfg.image_width
        
        if epoch == 0:
            # Initial setup for new view
            return self._initial_setup(
                embedding, means_3d, camtoworlds, Ks,
                width, height, view_idx, gt_pixels
            )
        else:
            # Regular training step
            return self._training_step(
                means_3d, camtoworlds, Ks,
                width, height, gt_pixels, epoch, view_idx
            )
    
    def _initial_setup(
        self,
        embedding: torch.Tensor,
        means_3d: torch.Tensor,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        view_idx: int,
        gt_pixels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Initial setup for a new view.
        
        Returns:
            Empty loss dictionary (no training in this step)
        """
        # Load pretrained models
        self._load_pretrained_models()
        self.encoder.eval()
        self.decoder.eval()
        
        # Forward pass with encoder
        with torch.no_grad():
            self.w_vectors = self.encoder(embedding)
            splats, _ = self.decoder(self.w_vectors, 0)
            splats["means"] = means_3d.squeeze(0) + splats["means"]
            # Render from all viewpoints for initial state
            
            self.render_all_views(
                splats=splats,
                prefix="initial",
                training_view=view_idx,
                step=0
            )
        
        # Setup optimizers for this view
        self.w_vectors = self.w_vectors.detach().requires_grad_()
        self.w_optimizer = self.decoder.setup_w_vector_optimizer(
            self.w_vectors,
            w_vector_lr=self.cfg.w_lr
        )
        
        mlp_optimizers = self.decoder.setup_optimizers(
            base_lr=self.cfg.base_lr,
            conditioning_lr=self.cfg.mlp_lr,
            w_vector_lr=0.0,
            embedding_lr=self.cfg.embedding_lr
        )
        self.mlp_optimizer = mlp_optimizers.get("conditioning")
        
        return {}
    
    def _training_step(
        self,
        means_3d: torch.Tensor,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        gt_pixels: torch.Tensor,
        epoch: int,
        view_idx: int
    ) -> Dict[str, float]:
        """
        Regular training step.
        
        Returns:
            Dictionary of loss values
        """
        self.decoder.train()
        
        # Forward pass
        splats, raw_outputs = self.decoder(self.w_vectors, epoch)
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
        
        # Optimization schedule
        halfway_point = self.cfg.max_epochs // 2
        
        if epoch == halfway_point and self.mlp_optimizer is not None:
            self.mlp_optimizer.zero_grad()
        
        if epoch <= halfway_point:
            # Optimize W vectors first
            if self.w_optimizer is not None:
                self.w_optimizer.step()
                self.w_optimizer.zero_grad()
        else:
            # Then optimize MLP
            if self.mlp_optimizer is not None:
                self.mlp_optimizer.step()
                self.mlp_optimizer.zero_grad()
        
        # Store splats and current means for final rendering
        self.current_splats = splats
        self.current_means_3d = means_3d
        
        return {k: v.item() for k, v in losses.items()}
    
    def run(self):
        """Main training loop."""
        print(f"\nStarting training for {self.cfg.max_epochs} epochs...")
        
        for view_idx, data in enumerate(self.trainloader, 1):
            print(f"\n{'='*50}")
            print(f"Training on view {view_idx}/{len(self.trainloader)}")
            print(f"{'='*50}")
            
            # Progress bar for epochs
            pbar = tqdm.tqdm(
                range(self.cfg.max_epochs),
                desc=f"Training View {view_idx}"
            )
            
            for epoch in pbar:
                # Training step
                losses = self.train_step(data, epoch, view_idx)
                
                # Update progress bar
                if losses:
                    loss_str = " ".join([
                        f"{k}: {v:.4f}" for k, v in losses.items()
                        if k != 'total'
                    ])
                    pbar.set_postfix_str(f"Loss: {losses.get('total', 0):.4f} | {loss_str}")
                
                self.global_step += 1
            
            # Save final outputs for this view from all camera angles
            if hasattr(self, 'current_splats'):
                with torch.no_grad():
                    self.decoder.eval()
                    
                    # The current_splats already have means added for the training view
                    # We need to render from all views with proper means adjustment
                    final_splats = self.current_splats.copy()
                    
                    # Render from all viewpoints for final state
                    self.render_all_views(
                        splats=final_splats,
                        prefix="final",
                        training_view=view_idx,
                        step=self.cfg.max_epochs - 1
                    )
        
        print("\nTraining completed!")
        print(f"Results saved to: {self.cfg.save_path}")
        print(f"Each training view has been rendered from all {len(self.all_cameras)} camera angles")


def create_config_from_args() -> TrainingConfig:
    """
    Create configuration from command line arguments.
    
    Returns:
        Training configuration object
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
        '--ply_file_path', type=str,
        default="pretrained_weights/averaged_model.ply",
        help='Path to base PLY model file'
    )
    parser.add_argument(
        '--save_path', type=str, default="results",
        help='Directory for saving results'
    )
    parser.add_argument(
        '--encoder_load_path', type=str,
        default="pretrained_weights/encoder_neutral_flame.pth",
        help='Path to pretrained encoder'
    )
    parser.add_argument(
        '--decoder_load_path', type=str,
        default="pretrained_weights/decoder_neutral_flame.pth",
        help='Path to pretrained decoder'
    )
    
    # Training parameters
    parser.add_argument(
        '--max_epochs', type=int, default=301,
        help='Number of training epochs per view'
    )
    parser.add_argument(
        '--sample_id', type=int, default=306,
        help='Sample ID for training'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_workers', type=int, default=1,
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
    
    # Optimization
    parser.add_argument(
        '--ga_step', type=int, default=1,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    # Network choices
    parser.add_argument(
        '--lpips_net', type=str, default="alex",
        choices=["alex", "vgg"],
        help='LPIPS network type'
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
    python scripts/inference_feedforward_full_guidance.py
    
    # Custom sample and epochs
    python scripts/inference_feedforward_full_guidance.py --sample_id 200 --max_epochs 1000
    
    # Custom learning rates
    python scripts/inference_feedforward_full_guidance.py --mlp_lr 2e-4 --w_lr 5e-5
    
    # With regularization
    python scripts/inference_feedforward_full_guidance.py --scale_reg 0.01 --pos_reg 0.001
    
    # Custom paths
    python scripts/inference_feedforward_full_guidance.py --data_root /path/to/data --save_path /path/to/results
    
    # Enable packed mode and sparse gradients
    python scripts/inference_feedforward_full_guidance.py --packed --sparse_grad
    
    # Different LPIPS network
    python scripts/inference_feedforward_full_guidance.py --lpips_net vgg
    """
    main()