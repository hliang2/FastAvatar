"""
Conditional Gaussian Splatting Model
=====================================
This module implements a conditional Gaussian splatting model with:
- Hypernet for parameter generation
- View-invariant face encoding
- Optimizable base Gaussian parameters
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp

from utils import load_ply_to_splats

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Hypernet(nn.Module):
    """
    Multi-layer perceptron for conditional Gaussian Splatting.
    
    Architecture:
        - Shared body network for feature extraction
        - Separate heads for different Gaussian parameters
        - Takes W vector and Gaussian embedding as input
    
    Args:
        w_dim: Dimension of W latent vector (default: 512)
        gaussian_embedding_dim: Dimension of gaussian embedding (default: 32)
        hidden_dim: Hidden dimension size (default: 256)
        body_layers: Number of layers in shared body (default: 6)
        head_layers: Number of layers in each head (default: 1)
        sh_degree: Spherical harmonics degree (default: 3)
    """
    
    def __init__(
        self,
        w_dim: int = 512,
        gaussian_embedding_dim: int = 32,
        hidden_dim: int = 256,
        body_layers: int = 6,
        head_layers: int = 1,
        sh_degree: int = 3,
    ):
        super().__init__()
        
        # Store dimensions
        self.input_dim = w_dim + gaussian_embedding_dim
        self.sh_dim = (sh_degree + 1) ** 2 * 3  # RGB channels with SH coefficients
        self.hidden_dim = hidden_dim
        
        # Build network components
        self.shared_body = self._build_shared_body(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=body_layers
        )
        
        # Create parameter-specific heads
        self.scale_head = self._build_head(hidden_dim, 3, head_layers)      # 3D scale
        self.rotation_head = self._build_head(hidden_dim, 4, head_layers)   # Quaternion
        self.color_head = self._build_head(hidden_dim, self.sh_dim, head_layers)  # SH coefficients
        self.opacity_head = self._build_head(hidden_dim, 1, head_layers)    # Opacity
        self.means_head = self._build_head(hidden_dim, 3, head_layers)      # Position deltas
        
        # Initialize weights for stability
        self._init_weights()
    
    def _build_shared_body(self, input_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
        """
        Build the shared body network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            
        Returns:
            Sequential network module
        """
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ])
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])
        
        return nn.Sequential(*layers)
    
    def _build_head(self, input_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
        """
        Build a head network for specific parameter prediction.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_layers: Number of layers
            
        Returns:
            Sequential network module
        """
        if num_layers == 1:
            # Single layer head
            return nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, output_dim)
            )
        
        # Multi-layer head (for future extensibility)
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, input_dim),
                nn.ReLU()
            ])
        layers.append(nn.Linear(input_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize network weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        w_vector: torch.Tensor, 
        gaussian_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            w_vector: Latent code tensor [batch_size, w_dim]
            gaussian_embedding: Gaussian embedding tensor [batch_size, gaussian_embedding_dim]
            
        Returns:
            Dictionary containing:
                - scale: Scale predictions
                - rotation: Rotation quaternion predictions
                - sh0: DC component of spherical harmonics
                - shN: Higher order spherical harmonics
                - means: Position delta predictions
                - opacity: Opacity predictions
        """
        # Concatenate inputs
        x = torch.cat([w_vector, gaussian_embedding], dim=1)
        
        # Extract shared features
        shared_features = self.shared_body(x)
        
        # Generate parameter predictions
        scale_output = self.scale_head(shared_features)
        rotation_output = self.rotation_head(shared_features)
        color_output = self.color_head(shared_features)
        opacity_output = self.opacity_head(shared_features)
        means_output = self.means_head(shared_features)
        
        # Format spherical harmonics coefficients
        sh0 = color_output[:, :3].view(-1, 1, 3)  # DC component
        
        if self.sh_dim > 3:
            shN = color_output[:, 3:].view(-1, (self.sh_dim // 3) - 1, 3)  # Higher orders
        else:
            shN = torch.zeros(-1, 0, 3, device=color_output.device)
        
        return {
            'scale': scale_output,
            'rotation': rotation_output,
            'sh0': sh0,
            'shN': shN,
            'means': means_output,
            'opacity': opacity_output.squeeze(-1)
        }


class CondGaussianSplatting(nn.Module):
    """
    Complete conditional Gaussian Splatting model.
    
    Combines pre-trained Gaussians with conditioning MLPs to enable
    dynamic modification of Gaussian parameters based on latent codes.
    
    Args:
        ply_path: Path to PLY file containing base Gaussians
        w_dim: Dimension of W latent vector (default: 512)
        gaussian_embedding_dim: Dimension of gaussian embedding (default: 32)
        hidden_dim: Hidden dimension size (default: 256)
        body_layers: Number of layers in shared body (default: 6)
        head_layers: Number of layers in each head (default: 1)
        sh_degree: Spherical harmonics degree (default: 3)
        gaussians_per_round: Batch size for processing Gaussians (default: 10144)
        optimize_base_gaussians: Whether to make base Gaussians trainable (default: True)
    """
    
    def __init__(
        self,
        ply_path: str,
        w_dim: int = 512,
        gaussian_embedding_dim: int = 32,
        hidden_dim: int = 256,
        body_layers: int = 6,
        head_layers: int = 1,
        sh_degree: int = 3,
        gaussians_per_round: int = 10144,
        optimize_base_gaussians: bool = True,
    ):
        super().__init__()
        
        # Load base Gaussians
        self.splats = load_ply_to_splats(ply_path).to(device)
        self.num_of_gaussians = len(self.splats["means"])
        
        # Initialize embeddings
        self.gaussian_embeddings = self._initialize_gaussian_embeddings(
            self.splats["means"], 
            gaussian_embedding_dim
        )
        
        # Configuration
        self.optimize_base_gaussians = optimize_base_gaussians
        self.gaussians_per_round = gaussians_per_round
        
        # Initialize conditioning network
        self.conditioning_mlp = Hypernet(
            w_dim=w_dim,
            gaussian_embedding_dim=gaussian_embedding_dim,
            hidden_dim=hidden_dim,
            body_layers=body_layers,
            head_layers=head_layers,
            sh_degree=sh_degree,
        )
        
        # Setup base Gaussian parameters
        self._setup_base_parameters()
    
    def _setup_base_parameters(self):
        """Initialize base Gaussian parameters as trainable or non-trainable."""
        if self.optimize_base_gaussians:
            # Make parameters trainable
            self.base_rotations = nn.Parameter(self.splats["quats"].clone())
            self.base_scales = nn.Parameter(self.splats["scales"].clone())
            self.base_sh0 = nn.Parameter(self.splats["sh0"].clone())
            self.base_shN = nn.Parameter(self.splats["shN"].clone())
            self.base_opacity = nn.Parameter(self.splats["opacities"].clone())
        else:
            # Keep as non-trainable buffers
            self.register_buffer('base_rotations', self.splats["quats"])
            self.register_buffer('base_scales', self.splats["scales"])
            self.register_buffer('base_sh0', self.splats["sh0"])
            self.register_buffer('base_shN', self.splats["shN"])
            self.register_buffer('base_opacity', self.splats["opacities"])
    
    def _initialize_gaussian_embeddings(
        self, 
        splat_means: torch.Tensor, 
        embedding_dim: int = 32
    ) -> nn.Embedding:
        """
        Initialize Gaussian embeddings with position encoding.
        
        Args:
            splat_means: Gaussian center positions
            embedding_dim: Embedding dimension
            
        Returns:
            Embedding layer with initialized weights
        """
        num_gaussians = splat_means.shape[0]
        embedding = nn.Embedding(num_gaussians, embedding_dim)
        
        # Normalize positions to [0,1]
        pos_min = splat_means.min(dim=0)[0]
        pos_max = splat_means.max(dim=0)[0]
        pos_range = pos_max - pos_min
        normalized_positions = (splat_means - pos_min) / pos_range
        
        # Initialize embedding with normalized coordinates
        embedding.weight.data[:, :3] = normalized_positions
        embedding.weight.data[:, 3:] = torch.randn(num_gaussians, embedding_dim - 3) * 0.02
        
        return embedding
    
    def get_base_gaussian_parameters(self) -> List[nn.Parameter]:
        """
        Get base Gaussian parameters for optimization.
        
        Returns:
            List of trainable base Gaussian parameters
        """
        if self.optimize_base_gaussians:
            return [
                self.base_rotations,
                self.base_scales,
                self.base_sh0,
                self.base_shN,
                self.base_opacity
            ]
        return []
    
    def get_conditioning_parameters(self) -> List[nn.Parameter]:
        """
        Get conditioning MLP parameters.
        
        Returns:
            List of conditioning network parameters
        """
        return list(self.conditioning_mlp.parameters())
    
    def get_gaussian_embedding_parameters(self) -> List[nn.Parameter]:
        """
        Get Gaussian embedding parameters.
        
        Returns:
            List of embedding parameters
        """
        return list(self.gaussian_embeddings.parameters())
    
    def setup_optimizers(
        self,
        base_lr: float = 1e-4,
        conditioning_lr: float = 1e-3,
        w_vector_lr: float = 1e-3,
        embedding_lr: float = 1e-4
    ) -> Dict[str, torch.optim.Optimizer]:
        """
        Setup separate optimizers for all trainable components.
        
        Args:
            base_lr: Learning rate for base Gaussian parameters
            conditioning_lr: Learning rate for conditioning network
            w_vector_lr: Learning rate for W vectors (unused here)
            embedding_lr: Learning rate for Gaussian embeddings
            
        Returns:
            Dictionary of optimizers
        """
        optimizers = {}
        
        # Conditioning network optimizer
        conditioning_params = self.get_conditioning_parameters()
        if conditioning_params:
            optimizers['conditioning'] = torch.optim.Adam(
                conditioning_params, 
                lr=conditioning_lr
            )
        
        # Gaussian embeddings optimizer
        embedding_params = self.get_gaussian_embedding_parameters()
        if embedding_params:
            optimizers['gaussian_embeddings'] = torch.optim.Adam(
                embedding_params, 
                lr=embedding_lr
            )
        
        # Base Gaussians optimizer (if trainable)
        base_params = self.get_base_gaussian_parameters()
        if base_params:
            optimizers['base_gaussians'] = torch.optim.Adam(
                base_params, 
                lr=base_lr
            )
        
        return optimizers
    
    def setup_w_vector_optimizer(
        self,
        w_vectors: nn.Parameter,
        w_vector_lr: float = 1e-3
    ) -> Optional[torch.optim.Optimizer]:
        """
        Setup optimizer for W vectors.
        
        Args:
            w_vectors: W vector parameters (from dataset)
            w_vector_lr: Learning rate for W vectors
            
        Returns:
            Optimizer for W vectors or None if not trainable
        """
        if w_vectors.requires_grad:
            return torch.optim.Adam([w_vectors], lr=w_vector_lr)
        
        print("Warning: W vectors do not require gradients. "
              "Ensure they are nn.Parameter with requires_grad=True")
        return None
    
    def setup_all_optimizers(
        self,
        w_vectors: nn.Parameter,
        base_lr: float = 1e-3,
        conditioning_lr: float = 1e-4,
        w_vector_lr: float = 1e-3,
        embedding_lr: float = 1e-4
    ) -> Dict[str, torch.optim.Optimizer]:
        """
        Setup all optimizers including W vectors.
        
        Args:
            w_vectors: W vector parameters from dataset
            base_lr: Learning rate for base Gaussian parameters
            conditioning_lr: Learning rate for conditioning network
            w_vector_lr: Learning rate for W vectors
            embedding_lr: Learning rate for Gaussian embeddings
            
        Returns:
            Dictionary containing all optimizers
        """
        # Get model optimizers
        optimizers = self.setup_optimizers(
            base_lr, 
            conditioning_lr, 
            w_vector_lr, 
            embedding_lr
        )
        
        # Add W vector optimizer
        w_optimizer = self.setup_w_vector_optimizer(w_vectors, w_vector_lr)
        if w_optimizer is not None:
            optimizers['w_vectors'] = w_optimizer
        
        return optimizers
    
    def setup_schedulers(
        self,
        optimizers: Dict[str, torch.optim.Optimizer],
        **scheduler_kwargs
    ) -> Dict[str, torch.optim.lr_scheduler._LRScheduler]:
        """
        Setup learning rate schedulers for all optimizers.
        
        Args:
            optimizers: Dictionary of optimizers
            **scheduler_kwargs: Additional scheduler arguments
            
        Returns:
            Dictionary containing schedulers
        """
        schedulers = {}
        
        # Default scheduler configuration
        mode = scheduler_kwargs.get('mode', 'min')
        factor = scheduler_kwargs.get('factor', 0.5)
        patience = scheduler_kwargs.get('patience', 10)
        
        for name, optimizer in optimizers.items():
            schedulers[name] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode=mode, 
                factor=factor, 
                patience=patience
            )
        
        return schedulers
    
    def setup_all_optimizers_and_schedulers(
        self,
        w_vectors: nn.Parameter,
        base_lr: float = 1e-4,
        conditioning_lr: float = 1e-3,
        w_vector_lr: float = 1e-3,
        embedding_lr: float = 1e-4,
        **scheduler_kwargs
    ) -> Tuple[Dict[str, torch.optim.Optimizer], Dict[str, torch.optim.lr_scheduler._LRScheduler]]:
        """
        Setup optimizers and schedulers in one call.
        
        Args:
            w_vectors: W vector parameters
            base_lr: Learning rate for base Gaussians
            conditioning_lr: Learning rate for conditioning network
            w_vector_lr: Learning rate for W vectors
            embedding_lr: Learning rate for embeddings
            **scheduler_kwargs: Additional scheduler arguments
            
        Returns:
            Tuple of (optimizers_dict, schedulers_dict)
        """
        # Setup optimizers
        optimizers = self.setup_all_optimizers(
            w_vectors, 
            base_lr, 
            conditioning_lr, 
            w_vector_lr, 
            embedding_lr
        )
        
        # Setup schedulers
        schedulers = self.setup_schedulers(optimizers, **scheduler_kwargs)
        
        return optimizers, schedulers
    
    def forward(
        self, 
        w_vector: torch.Tensor, 
        step: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass that modifies base Gaussians according to conditioning.
        
        Args:
            w_vector: Latent code [batch_size, w_dim]
            step: Current training step (for logging)
            
        Returns:
            Tuple of:
                - Modified Gaussian parameters dictionary
                - Raw network outputs dictionary
        """
        # Update configuration flags
        update_positions = True
        update_scales = True
        update_rotations = True
        update_appearance = True
        update_opacity = True
        
        # Get all gaussian indices
        active_indices = torch.arange(self.num_of_gaussians, device=device)
        
        # Clone base parameters
        rotations = self.base_rotations.clone()
        scales = self.base_scales.clone()
        sh0 = self.base_sh0.clone()
        shN = self.base_shN.clone()
        opacity = self.base_opacity.clone()
        
        # Initialize means storage
        all_means_deltas = []
        
        # Process gaussians in batches
        for g_start in range(0, len(active_indices), self.gaussians_per_round):
            g_end = min(g_start + self.gaussians_per_round, len(active_indices))
            batch_indices = active_indices[g_start:g_end]
            
            # Get embeddings for this batch
            g_embeddings = self.gaussian_embeddings(batch_indices)
            
            # Expand w_vector to match gaussian batch size
            batch_size_g = g_end - g_start
            expanded_w = w_vector.expand(batch_size_g, -1)
            
            # Get predictions
            output = self.conditioning_mlp(expanded_w, g_embeddings)
            
            # Apply updates
            if update_positions:
                means_delta = torch.tanh(output["means"]) * 0.1
                all_means_deltas.append(means_delta)
            
            if update_scales:
                scales[batch_indices] = self.base_scales[batch_indices] + output["scale"]
            
            if update_rotations:
                rotations[batch_indices] = self.base_rotations[batch_indices] + output["rotation"]
            
            if update_appearance:
                sh0[batch_indices] = self.base_sh0[batch_indices] + output["sh0"]
                if output["shN"].shape[1] > 0:
                    shN[batch_indices] = self.base_shN[batch_indices] + output["shN"]
            
            if update_opacity:
                opacity[batch_indices] = self.base_opacity[batch_indices] + output["opacity"]
        
        # Combine all means deltas
        means = torch.cat(all_means_deltas, dim=0) if all_means_deltas else torch.zeros_like(self.splats["means"])
        
        # Prepare output dictionaries
        modified_params = {
            'means': means,
            'quats': rotations,
            'scales': scales,
            'sh0': sh0,
            'shN': shN,
            'opacities': opacity
        }
        
        raw_outputs = {
            "raw_means": means,
            "raw_opacities": output["opacity"],
            "raw_scales": output["scale"],
            "raw_rotations": output["rotation"]
        }
        
        return modified_params, raw_outputs


class ViewInvariantEncoder(nn.Module):
    """
    Pose-invariant encoder using InsightFace pretrained models.
    
    Maps face images to specific 512-dimensional latent codes while
    maintaining invariance to pose variations.
    
    Args:
        target_dim: Target latent dimension (default: 512)
        model_name: InsightFace model name (default: 'buffalo_l')
        device: Device to run on (default: 'cuda')
    """
    
    def __init__(
        self,
        target_dim: int = 512,
        model_name: str = 'buffalo_l',
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.device = device
        self.target_dim = target_dim
        
        # InsightFace output dimension
        insightface_dim = 512
        
        # Projection network architecture
        self.projector = nn.Sequential(
            # Layer 1: 512 -> 1024
            nn.Linear(insightface_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # Layer 2: 1024 -> 768
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # Layer 3: 768 -> 512
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            # Output layer: 512 -> target_dim
            nn.Linear(512, target_dim)
        )
        
        # Learnable temperature for scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection network weights."""
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_insightface_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input images [B, 3, H, W] in range [0, 1]
            return_insightface_features: Whether to return original features
        
        Returns:
            Projected embeddings [B, target_dim] or tuple with InsightFace features
        """
        # L2 normalize InsightFace embeddings
        insightface_features = F.normalize(x, p=2, dim=1)
        
        # Project to target space
        projected = self.projector(insightface_features)
        
        # Apply temperature scaling
        projected = projected * self.temperature
        
        if return_insightface_features:
            return projected, insightface_features
        
        return projected