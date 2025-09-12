"""
Encoder Training Script for View-Invariant Face Representations
===============================================================
This module provides training capabilities for the ViewInvariantEncoder
to learn optimal W vectors from face embeddings.
"""

import argparse
import os
import time
import pdb
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm

from dataset import GaussianFaceEncoderDataset
from model import CondGaussianSplatting, ViewInvariantEncoder
from utils import set_random_seed

# Suppress warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class EncoderTrainingConfig:
    """Configuration for encoder training."""
    
    # Data paths
    data_root: str = "data"
    max_subjects: int = 417
    decoder_load_path: str = "pretrained_weights/decoder_neutral_flame.pth"
    save_path: str = "checkpoints"
    
    # Training parameters
    max_epochs: int = 400
    batch_size: int = 64
    num_workers: int = 4
    
    # Learning rates
    encoder_lr: float = 1e-4
    
    # Loss weights
    mse_weight: float = 0.1
    cosine_weight: float = 1.0
    contrastive_weight: float = 0.0  # Disabled by default
    
    # Contrastive loss parameters
    contrastive_temperature: float = 0.07
    
    # Optimization
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    gradient_clip_norm: float = 1.0
    
    # Logging and saving
    log_every: int = 100
    save_every: int = 50
    model_save_steps: List[int] = field(default_factory=lambda: [50, 100, 200, 300, 400])
    
    # Random seed
    seed: int = 42


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning view-invariant representations."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings: torch.Tensor, ids: List[int]) -> torch.Tensor:
        """
        Compute contrastive loss for view-invariant learning.
        
        Args:
            embeddings: [batch_size, latent_dim] predicted embeddings
            ids: List of identity IDs
            
        Returns:
            Contrastive loss value
        """
        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create label matrix (1 if same identity, 0 otherwise)
        labels = torch.zeros(batch_size, batch_size, device=embeddings.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if ids[i] == ids[j]:
                    labels[i, j] = 1.0
        
        # Mask out diagonal
        mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Compute InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            # Find positive samples (same identity, different view)
            positive_mask = (labels[i] == 1) & (~mask[i])
            if positive_mask.sum() == 0:
                continue
                
            # Compute InfoNCE loss for this sample
            positive_sim = exp_sim[i][positive_mask].sum()
            all_sim = exp_sim[i].sum()
            
            if all_sim > 0:
                loss += -torch.log(positive_sim / all_sim)
                valid_samples += 1
        
        return loss / max(valid_samples, 1)


class EncoderTrainer:
    """
    Trainer class for ViewInvariantEncoder.
    """
    
    def __init__(self, config: EncoderTrainingConfig):
        """
        Initialize the encoder trainer.
        
        Args:
            config: Training configuration
        """
        self.cfg = config
        
        # Set random seed
        set_random_seed(config.seed)
        
        # Initialize models
        self._initialize_models()
        
        # Load pretrained decoder and W vectors
        self._load_pretrained_decoder()
        
        # Setup dataset and dataloader
        self._setup_data()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Initialize losses
        self._initialize_losses()
        
        # Create output directories
        self._setup_directories()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        print(f"Encoder trainer initialized with {len(self.trainset)} training samples")
    
    def _initialize_models(self):
        """Initialize encoder model."""
        print("Initializing encoder...")
        
        self.encoder = ViewInvariantEncoder().to(device)
        
        # We don't need the decoder for training, but we need its W vectors
        self.decoder = CondGaussianSplatting(
            ply_path="pretrained_weights/averaged_model.ply"  # Placeholder
        ).to(device)
        
        print("Encoder initialized")
    
    def _load_pretrained_decoder(self):
        """Load pretrained decoder and extract W vectors."""
        print("Loading pretrained decoder and W vectors...")
        
        if not os.path.exists(self.cfg.decoder_load_path):
            raise FileNotFoundError(f"Decoder checkpoint not found: {self.cfg.decoder_load_path}")
        
        checkpoint = torch.load(self.cfg.decoder_load_path, map_location=device)
        
        # Load decoder state (we need it for W vectors)
        self.decoder.load_state_dict(checkpoint["model_dict"])
        
        # Extract W vectors and mapping
        self.w_vectors = checkpoint["w_vectors"].to(device)
        self.w_ids_to_idx = checkpoint["w_ids_to_idx"]

        print(f"Loaded W vectors: {self.w_vectors.shape}")
        print(f"Available identities: {list(self.w_ids_to_idx.keys())}")
    
    def _setup_data(self):
        """Setup dataset and dataloader."""
        print("Setting up dataset...")
        
        # Initialize training dataset
        self.trainset = GaussianFaceEncoderDataset(
            data_root=self.cfg.data_root,
            seed=self.cfg.seed,
            max_subjects=int(self.cfg.max_subjects)
        )
        
        # Create dataloader
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
        
        print(f"Training dataset loaded: {len(self.trainset)} samples")
        print(f"Identities in dataset: {self.trainset.get_data_statistics()['num_identities']}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # We can choose to train the entire encoder or just the projector
        # For this version, we'll train the entire encoder
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.encoder_lr
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.cfg.scheduler_factor,
            patience=self.cfg.scheduler_patience,
            verbose=True
        )
    
    def _initialize_losses(self):
        """Initialize loss functions."""
        self.contrastive_loss = ContrastiveLoss(temperature=self.cfg.contrastive_temperature)
        self.mse_loss = nn.MSELoss()
    
    def _setup_directories(self):
        """Create output directories."""
        # Create experiment name
        exp_name = f"encoder_lr{self.cfg.encoder_lr}_mse{self.cfg.mse_weight}_cos{self.cfg.cosine_weight}"
        if self.cfg.contrastive_weight > 0:
            exp_name += f"_cont{self.cfg.contrastive_weight}"
        
        # Create directory paths
        self.dirs = {
            'base': Path(self.cfg.save_path) / exp_name,
            'checkpoints': Path(self.cfg.save_path) / exp_name / 'checkpoints',
            'logs': Path(self.cfg.save_path) / exp_name / 'logs',
        }
        
        # Create directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directories created under: {self.dirs['base']}")
    
    def compute_losses(
        self,
        predicted_codes: torch.Tensor,
        target_codes: torch.Tensor,
        image_ids: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            predicted_codes: Predicted W vectors from encoder
            target_codes: Target W vectors from pretrained decoder
            image_ids: List of identity IDs for contrastive loss
            
        Returns:
            Dictionary of loss values
        """
        losses = {}
        
        # MSE loss with target latent codes
        if self.cfg.mse_weight > 0:
            losses['mse'] = self.mse_loss(predicted_codes, target_codes)
        
        # Cosine similarity loss
        if self.cfg.cosine_weight > 0:
            predicted_norm = F.normalize(predicted_codes, p=2, dim=1)
            target_norm = F.normalize(target_codes, p=2, dim=1)
            cosine_sim = (predicted_norm * target_norm).sum(dim=1).mean()
            losses['cosine'] = 1.0 - cosine_sim
        
        # Contrastive loss for view invariance
        if self.cfg.contrastive_weight > 0:
            losses['contrastive'] = self.contrastive_loss(predicted_codes, image_ids)
        
        # Combined loss
        total_loss = 0.0
        if 'mse' in losses:
            total_loss += losses['mse'] * self.cfg.mse_weight
        if 'cosine' in losses:
            total_loss += losses['cosine'] * self.cfg.cosine_weight
        if 'contrastive' in losses:
            total_loss += losses['contrastive'] * self.cfg.contrastive_weight
        
        losses['total'] = total_loss
        
        return losses
    
    def train_step(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            data: Batch data dictionary
            
        Returns:
            Dictionary of loss values
        """
        # Extract data
        image_ids = data["id"]
        embeddings = data["embedding"].to(device)
        
        # Get target W vectors for this batch
        try:
            w_indices = [self.w_ids_to_idx[id_] for id_ in image_ids]
            target_codes = self.w_vectors[w_indices]
        except KeyError as e:
            print(f"Warning: Identity {e} not found in W vector mapping, skipping batch")
            return {'total': 0.0}
        
        # Forward pass
        predicted_codes = self.encoder(embeddings)
        
        # Compute losses
        losses = self.compute_losses(predicted_codes, target_codes, image_ids)
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        if self.cfg.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(),
                max_norm=self.cfg.gradient_clip_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {k: v.item() for k, v in losses.items()}
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the encoder on the training set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.encoder.eval()
        
        total_mse = 0.0
        total_cosine = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data in self.trainloader:
                image_ids = data["id"]
                embeddings = data["embedding"].to(device)
                
                # Get target W vectors
                try:
                    w_indices = [self.w_ids_to_idx[id_] for id_ in image_ids]
                    target_codes = self.w_vectors[w_indices]
                except KeyError:
                    continue
                
                # Forward pass
                predicted_codes = self.encoder(embeddings)
                
                # Compute metrics
                mse = self.mse_loss(predicted_codes, target_codes)
                
                predicted_norm = F.normalize(predicted_codes, p=2, dim=1)
                target_norm = F.normalize(target_codes, p=2, dim=1)
                cosine_sim = (predicted_norm * target_norm).sum(dim=1).mean()
                
                total_mse += mse.item()
                total_cosine += cosine_sim.item()
                num_batches += 1
                
                # Limit evaluation for efficiency
                if num_batches >= 20:
                    break
        
        self.encoder.train()
        
        return {
            'eval_mse': total_mse / max(num_batches, 1),
            'eval_cosine_sim': total_cosine / max(num_batches, 1),
        }
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "model_dict": self.encoder.state_dict(),
            "optimizer_dict": self.optimizer.state_dict(),
            "scheduler_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "config": self.cfg,
        }
        
        # Save latest checkpoint
        latest_path = self.dirs['checkpoints'] / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save epoch-specific checkpoint
        epoch_path = self.dirs['checkpoints'] / f"encoder_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, epoch_path)
        
        # Also save just the model dict for easy loading
        model_only_path = self.dirs['checkpoints'] / f"encoder_model_epoch_{epoch:04d}.pth"
        torch.save({"model_dict": self.encoder.state_dict()}, model_only_path)
        
        print(f"Checkpoint saved: {epoch_path}")
    
    def run(self):
        """Main training loop."""
        print(f"\nStarting encoder training for {self.cfg.max_epochs} epochs...")
        print(f"Batch size: {self.cfg.batch_size}")
        print(f"Learning rate: {self.cfg.encoder_lr}")
        print(f"Loss weights - MSE: {self.cfg.mse_weight}, Cosine: {self.cfg.cosine_weight}, Contrastive: {self.cfg.contrastive_weight}")
        
        self.encoder.train()
        
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
                losses = self.train_step(data)
                
                if losses['total'] > 0:  # Skip if batch was invalid
                    epoch_losses.append(losses['total'])
                
                # Update progress bar
                if losses and losses['total'] > 0:
                    loss_str = " ".join([
                        f"{k}: {v:.4f}" for k, v in losses.items()
                        if k != 'total'
                    ])
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix_str(f"Loss: {losses['total']:.4f} | LR: {current_lr:.2e} | {loss_str}")
                
                self.global_step += 1
            
            # Compute average epoch loss
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
            else:
                avg_loss = 0.0
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")
            
            # Learning rate scheduling
            self.scheduler.step(avg_loss)
            
            # Evaluation
            if (epoch + 1) % 20 == 0:
                eval_metrics = self.evaluate()
                print(f"Eval - MSE: {eval_metrics['eval_mse']:.4f}, "
                      f"Cosine Sim: {eval_metrics['eval_cosine_sim']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) in self.cfg.model_save_steps or (epoch + 1) % self.cfg.save_every == 0:
                self.save_checkpoint(epoch + 1)
            
            self.current_epoch = epoch + 1
        
        # Final checkpoint
        self.save_checkpoint(self.current_epoch)
        print("\nEncoder training completed!")
        print(f"Results saved to: {self.dirs['base']}")


def create_config_from_args() -> EncoderTrainingConfig:
    """
    Create configuration from command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Encoder Training for View-Invariant Face Representations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        '--data_root', type=str, default="data",
        help='Path to the dataset root directory'
    )
    parser.add_argument(
        '--max_subjects', type=str, default="417",
        help='Max subjects for quicker training/validation, should be consistent with decoder'
    )
    parser.add_argument(
        '--decoder_load_path', type=str,
        default="pretrained_weights/decoder_neutral_flame.pth",
        help='Path to pretrained decoder checkpoint'
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
        '--batch_size', type=int, default=64,
        help='Batch size for training'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers'
    )
    
    # Learning rate
    parser.add_argument(
        '--encoder_lr', type=float, default=1e-4,
        help='Learning rate for encoder'
    )
    
    # Loss weights
    parser.add_argument(
        '--mse_weight', type=float, default=0.1,
        help='Weight for MSE loss'
    )
    parser.add_argument(
        '--cosine_weight', type=float, default=1.0,
        help='Weight for cosine similarity loss'
    )
    parser.add_argument(
        '--contrastive_weight', type=float, default=0.0,
        help='Weight for contrastive loss'
    )
    
    # Contrastive loss parameters
    parser.add_argument(
        '--contrastive_temperature', type=float, default=0.07,
        help='Temperature for contrastive loss'
    )
    
    # Optimization
    parser.add_argument(
        '--scheduler_patience', type=int, default=5,
        help='Patience for learning rate scheduler'
    )
    parser.add_argument(
        '--scheduler_factor', type=float, default=0.5,
        help='Factor for learning rate reduction'
    )
    parser.add_argument(
        '--gradient_clip_norm', type=float, default=1.0,
        help='Gradient clipping norm'
    )
    
    # Logging
    parser.add_argument(
        '--log_every', type=int, default=1000,
        help='Logging frequency (steps)'
    )
    parser.add_argument(
        '--save_every', type=int, default=1000,
        help='Save frequency (epochs)'
    )
    parser.add_argument(
        '--model_save_steps', type=lambda s: [int(item) for item in s.split(',')],
        default="50",
        help='Comma-separated list of epochs to save model'
    )
    
    # Random seed
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = EncoderTrainingConfig(**vars(args))
    
    return config


def main():
    """Main entry point for encoder training."""
    # Parse arguments and create config
    config = create_config_from_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("Encoder Training Configuration")
    print("="*60)
    for key, value in vars(config).items():
        print(f"{key:25s}: {value}")
    print("="*60 + "\n")
    
    # Create and run trainer
    trainer = EncoderTrainer(config)
    trainer.run()


if __name__ == "__main__":
    """
    Usage examples:
    
    # Basic training
    python scripts/train_encoder.py --data_root data --decoder_load_path checkpoints/decoder.pth
    
    # Custom learning rate and loss weights
    python scripts/train_encoder.py --encoder_lr 2e-4 --mse_weight 0.2 --cosine_weight 0.8
    
    # Enable contrastive loss
    python scripts/train_encoder.py --contrastive_weight 0.1 --contrastive_temperature 0.1
    
    # Custom batch size and epochs
    python scripts/train_encoder.py --batch_size 128 --max_epochs 800
    
    # Custom save schedule
    python scripts/train_encoder.py --model_save_steps "100,200,400,600,800"
    """
    main()