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
    
class DINOv2Encoder(nn.Module):
    """
    Improved encoder with proper rotation representation and residual point prediction
    """
    
    def __init__(
        self,
        dino_model='dinov2_vitb14',
        max_points=10000,
        hidden_dim=1024,
        num_layers=3,
        dropout=0.1,
        freeze_dino=False,
        use_rotation_6d=True,  # Use 6D rotation representation
        predict_point_residuals=True,  # Predict residuals from mean shape
    ):
        super().__init__()
        
        self.max_points = max_points
        self.use_rotation_6d = use_rotation_6d
        self.predict_point_residuals = predict_point_residuals
        
        # Load DINOv2 model
        self.dino = torch.hub.load('facebookresearch/dinov2', dino_model)
        
        # Get feature dimension
        if 'vits' in dino_model:
            self.feat_dim = 384
        elif 'vitb' in dino_model:
            self.feat_dim = 768
        elif 'vitl' in dino_model:
            self.feat_dim = 1024
        elif 'vitg' in dino_model:
            self.feat_dim = 1536
        
        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False
        
        # Projection head for 3D points (residuals or full)
        self.points_projector = self._build_projector(
            self.feat_dim,
            hidden_dim,
            max_points * 3,
            num_layers,
            dropout
        )
        
        # Projection head for camera pose
        # 6D rotation (6) + translation (3) = 9 parameters
        # or quaternion (4) + translation (3) = 7 parameters
        pose_dim = 9 if use_rotation_6d else 7
        self.pose_projector = self._build_projector(
            self.feat_dim,
            hidden_dim,
            pose_dim,
            num_layers,
            dropout
        )
        
        # Learnable mean shape (optional)
        if predict_point_residuals:
            self.register_buffer('mean_shape', torch.zeros(max_points, 3))
            self.learn_mean_shape = nn.Parameter(torch.zeros(max_points, 3))
    
    def _build_projector(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
            if i < num_layers - 1:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        return nn.Sequential(*layers)
    
    def rotation_6d_to_matrix(self, d6):
        """
        Convert 6D rotation representation to rotation matrix.
        Based on Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"
        
        Args:
            d6: (B, 6) tensor
        
        Returns:
            R: (B, 3, 3) rotation matrix
        """
        a1, a2 = d6[..., :3], d6[..., 3:]
        
        # Gram-Schmidt orthogonalization
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        
        return torch.stack([b1, b2, b3], dim=-2)
    
    def quaternion_to_matrix(self, quaternion):
        """
        Convert quaternion to rotation matrix.
        
        Args:
            quaternion: (B, 4) tensor (w, x, y, z)
        
        Returns:
            R: (B, 3, 3) rotation matrix
        """
        # Normalize quaternion
        quaternion = F.normalize(quaternion, dim=-1)
        
        w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
        
        # Convert to rotation matrix
        R = torch.zeros(quaternion.shape[0], 3, 3, device=quaternion.device)
        
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y * z - w * x)
        
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
        
        return R
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Extract features with DINOv2
        features = self.dino.forward_features(x)
        
        # Use CLS token for both predictions
        cls_token = features['x_norm_clstoken']  # (B, feat_dim)
        
        # Predict camera pose
        pose_params = self.pose_projector(cls_token)  # (B, 9) or (B, 7)
        
        if self.use_rotation_6d:
            rotation_6d = pose_params[..., :6]
            translation = pose_params[..., 6:9]
            rotation_matrix = self.rotation_6d_to_matrix(rotation_6d)
        else:
            quaternion = pose_params[..., :4]
            translation = pose_params[..., 4:7]
            rotation_matrix = self.quaternion_to_matrix(quaternion)
        
        # Construct camera-to-world matrix
        c2w = torch.zeros(batch_size, 4, 4, device=x.device)
        c2w[:, :3, :3] = rotation_matrix
        c2w[:, :3, 3] = translation
        c2w[:, 3, 3] = 1.0
        
        # Predict 3D points
        points_output = self.points_projector(cls_token)  # (B, max_points * 3)
        points_output = points_output.view(batch_size, self.max_points, 3)
        
        if self.predict_point_residuals:
            # Add residuals to mean shape
            mean_shape = self.mean_shape + self.learn_mean_shape
            points = mean_shape.unsqueeze(0) + points_output * 0.1  # Scale residuals
        else:
            points = points_output
        
        return {
            'points': points,
            'camera_pose': c2w,  # Full 4x4 matrix
            'camera_pose_flat': c2w[:, :3, :].reshape(batch_size, -1),  # For loss computation
            'rotation_matrix': rotation_matrix,
            'translation': translation,
        }


class EncoderWithLoss(nn.Module):
    """Improved loss computation with proper rotation loss"""
    
    def __init__(
        self,
        encoder,
        points_weight=1.0,
        rotation_weight=1.0,
        translation_weight=1.0,
        use_geodesic_rotation_loss=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.points_weight = points_weight
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.use_geodesic_rotation_loss = use_geodesic_rotation_loss
    
    def geodesic_loss(self, R_pred, R_gt):
        """
        Compute geodesic distance between rotation matrices.
        This is the angle of rotation between two rotation matrices.
        """
        # Compute R_pred @ R_gt.T
        R_diff = torch.bmm(R_pred, R_gt.transpose(1, 2))
        
        # Compute trace
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        
        # Clamp to avoid numerical issues with arccos
        trace = torch.clamp(trace, min=-1.0 + 1e-6, max=3.0 - 1e-6)
        
        # Compute angle
        angle = torch.acos((trace - 1.0) / 2.0)
        
        return angle.mean()
    
    def forward(self, batch, training=True):
        # Get predictions
        predictions = self.encoder(batch['image'])
        
        outputs = {
            'points_pred': predictions['points'],
            'rotation_pred': predictions['rotation_matrix'],
            'translation_pred': predictions['translation'],
        }
        
        if training:
            losses = {}
            
            # Points loss
            points_loss = F.mse_loss(
                predictions['points'],
                batch['points']  # or batch['point_residuals'] if using residuals
            )
            losses['points'] = points_loss * self.points_weight
            
            # Extract ground truth rotation and translation
            gt_rotation = batch['rotation_matrix']  # (B, 3, 3)
            gt_translation = batch['translation']    # (B, 3)
            
            # Rotation loss
            if self.use_geodesic_rotation_loss:
                rotation_loss = self.geodesic_loss(
                    predictions['rotation_matrix'],
                    gt_rotation
                )
            else:
                rotation_loss = F.mse_loss(
                    predictions['rotation_matrix'],
                    gt_rotation
                )
            losses['rotation'] = rotation_loss * self.rotation_weight
            
            # Translation loss
            translation_loss = F.mse_loss(
                predictions['translation'],
                gt_translation
            )
            losses['translation'] = translation_loss * self.translation_weight
            
            # Total loss
            losses['total'] = losses['points'] + losses['rotation'] + losses['translation']
            
            outputs['losses'] = losses
        
        return outputs


def create_model(
    dino_model='dinov2_vitb14',
    max_points=10144,
    hidden_dim=1024,
    num_layers=3,
    dropout=0.1,
    freeze_dino=False,
    use_rotation_6d=True,
    predict_point_residuals=True,
    points_weight=1.0,
    rotation_weight=1.0,
    translation_weight=1.0,
    use_geodesic_rotation_loss=True,
):
    """Factory function to create the improved model"""
    
    encoder = DINOv2Encoder(
        dino_model=dino_model,
        max_points=max_points,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        freeze_dino=freeze_dino,
        use_rotation_6d=use_rotation_6d,
        predict_point_residuals=predict_point_residuals,
    )
    
    model = EncoderWithLoss(
        encoder=encoder,
        points_weight=points_weight,
        rotation_weight=rotation_weight,
        translation_weight=translation_weight,
        use_geodesic_rotation_loss=use_geodesic_rotation_loss,
    )
    
    return model