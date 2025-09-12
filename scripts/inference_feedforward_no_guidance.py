"""
Feedforward Inference Script for Gaussian Splatting
===================================================
This script performs end-to-end inference from a single image:
1. Extracts face embedding from image
2. Uses encoder to predict W vector from embedding
3. Uses DINO model to extract 3D point means
4. Uses decoder to generate Gaussian splats
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import json
from torchvision.utils import save_image
import imageio.v2 as imageio

from gsplat.rendering import rasterization
from gsplat.utils import save_ply

from dataset import SingleImageDataset
from model import CondGaussianSplatting, ViewInvariantEncoder, create_model
from utils import set_random_seed

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeedforwardInferenceEngine:
    """
    End-to-end inference engine for Gaussian Splatting from single image.
    """
    
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        dino_path: str,
        device: str = 'cuda'
    ):
        """
        Initialize the inference engine.
        
        Args:
            encoder_path: Path to trained encoder checkpoint
            decoder_path: Path to trained decoder checkpoint
            dino_path: Path to trained DINO model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device)
        
        # Load models
        print("Loading models...")
        self._load_encoder(encoder_path)
        self._load_decoder(decoder_path)
        self._load_dino_model(dino_path)
        
        print(f"All models loaded on {self.device}")
    
    def _load_encoder(self, encoder_path: str):
        """Load the trained encoder model."""
        print(f"Loading encoder from {encoder_path}")
        
        self.encoder = ViewInvariantEncoder().to(self.device)
        
        checkpoint = torch.load(encoder_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["model_dict"])
        self.encoder.eval()
        
        print("Encoder loaded successfully")
    
    def _load_decoder(self, decoder_path: str):
        """Load the trained decoder model."""
        print(f"Loading decoder from {decoder_path}")
        
        # Initialize decoder with placeholder PLY path
        self.decoder = CondGaussianSplatting(
            ply_path="pretrained_weights/averaged_model.ply"  # Will be overridden
        ).to(self.device)
        
        checkpoint = torch.load(decoder_path, map_location=self.device)
        self.decoder.load_state_dict(checkpoint["model_dict"])
        self.decoder.eval()
        
        print("Decoder loaded successfully")
    
    def _load_dino_model(self, dino_path: str):
        """Load the trained DINO model."""
        print(f"Loading DINO model from {dino_path}")
        
        # Load DINO model using your existing function
        self.dino_model, self.dino_metadata = self._load_dino_checkpoint(dino_path)
        
        print("DINO model loaded successfully")
    
    def _load_dino_checkpoint(self, checkpoint_path: str):
        """Load DINO model checkpoint (adapted from your code)."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get experiment directory
        exp_dir = Path(checkpoint_path).parent.parent
        
        # Load metadata
        metadata_path = exp_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            print("Warning: metadata.json not found, using default values")
            metadata = {
                'max_points': 10144,
                'dino_model': 'dinov2_vitb14',
                'use_rotation_6d': True,
                'predict_point_residuals': True,
                'mean_shape': None
            }
        
        # Load training args if available
        args_path = exp_dir / 'args.json'
        if args_path.exists():
            with open(args_path, 'r') as f:
                training_args = json.load(f)
        else:
            training_args = {}
        
        # Create model with same configuration
        model = create_model(
            dino_model=metadata.get('dino_model', 'dinov2_vitb14'),
            max_points=metadata.get('max_points', 10144),
            hidden_dim=training_args.get('hidden_dim', 1024),
            num_layers=training_args.get('num_layers', 3),
            dropout=0.0,
            freeze_dino=False,
            use_rotation_6d=metadata.get('use_rotation_6d', True),
            predict_point_residuals=metadata.get('predict_point_residuals', True),
            points_weight=1.0,
            rotation_weight=1.0,
            translation_weight=1.0,
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set mean shape if available
        if metadata.get('mean_shape') is not None:
            mean_shape = torch.tensor(metadata['mean_shape'], dtype=torch.float32)
            model.encoder.mean_shape = mean_shape.to(self.device)
        
        model = model.to(self.device)
        model.eval()
        
        return model, metadata
    
    @torch.no_grad()
    def predict_from_image(self, image_path: str) -> Dict:
        """
        Run end-to-end prediction from single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary containing all predictions
        """
        # Load and preprocess image
        dataset = SingleImageDataset(image_path)
        data = dataset[0]  # Get the single sample
        
        # Move data to device
        embedding = data["embedding"].unsqueeze(0).to(self.device)
        dino_image = data["dino_image"].unsqueeze(0).to(self.device)
        raw_image = data["raw_image"]
        
        # Step 1: Get W vector from encoder
        print("Predicting W vector from face embedding...")
        w_vector = self.encoder(embedding)  # [1, w_dim]
        
        # Step 2: Get 3D points from DINO
        print("Predicting 3D points from DINO model...")
        dino_predictions = self.dino_model.encoder(dino_image)
        points_3d = dino_predictions['points'][0]  # Remove batch dimension [N, 3]
        
        # Step 3: Generate Gaussian splats from decoder
        print("Generating Gaussian splats...")
        splats, raw_outputs = self.decoder(w_vector, step=0)
        
        # Step 4: Update splat means with DINO predictions
        splats["means"] = points_3d + splats["means"]
        
        # Compile results
        results = {
            'w_vector': w_vector,
            'dino_points': points_3d,
            'dino_predictions': dino_predictions,
            'splats': splats,
            'raw_outputs': raw_outputs,
            'raw_image': raw_image,
            'image_path': image_path,
        }
        
        return results

    
    def save_results(
        self,
        results: Dict,
        output_dir: str,
        save: bool = True
    ):
        """
        Save inference results to disk.
        
        Args:
            results: Results dictionary from predict_from_image
            output_dir: Output directory
            render_novel_views: Whether to render novel views
            save_ply: Whether to save PLY file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save W vector
        np.save(output_path / 'w_vector.npy', results['w_vector'].cpu().numpy())
        
        # Save DINO points
        np.save(output_path / 'dino_points.npy', results['dino_points'].cpu().numpy())
        
        # Save camera predictions from DINO
        if 'camera_pose' in results['dino_predictions']:
            np.save(output_path / 'dino_camera_pose.npy', 
                   results['dino_predictions']['camera_pose'].cpu().numpy())
        
        # Save PLY file
        if save:
            ply_path = str(output_path / 'splats.ply')
            save_ply(results['splats'], ply_path, None)
            print(f"PLY saved: {ply_path}")

        # Save summary
        summary = {
            'image_path': results['image_path'],
            'w_vector_shape': list(results['w_vector'].shape),
            'dino_points_shape': list(results['dino_points'].shape),
            'num_gaussians': int(results['splats']['means'].shape[0]),
            'splat_statistics': {
                'means_std': float(results['splats']['means'].std().item()),
                'scales_mean': float(torch.exp(results['splats']['scales']).mean().item()),
                'opacities_mean': float(torch.sigmoid(results['splats']['opacities']).mean().item()),
            }
        }
        
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Feedforward Gaussian Splatting Inference from Single Image',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--image', type=str, required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--encoder_checkpoint', type=str, required=True,
        help='Path to trained encoder checkpoint'
    )
    parser.add_argument(
        '--decoder_checkpoint', type=str, required=True,
        help='Path to trained decoder checkpoint'
    )
    parser.add_argument(
        '--dino_checkpoint', type=str, required=True,
        help='Path to trained DINO model checkpoint'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_dir', type=str, default='./results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--save_ply', action='store_true', default=True,
        help='Save PLY file'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Validation
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    for checkpoint in [args.encoder_checkpoint, args.decoder_checkpoint, args.dino_checkpoint]:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize inference engine
    print("Initializing inference engine...")
    engine = FeedforwardInferenceEngine(
        encoder_path=args.encoder_checkpoint,
        decoder_path=args.decoder_checkpoint,
        dino_path=args.dino_checkpoint,
        device=str(device)
    )
    
    # Run inference
    print(f"\nRunning inference on: {args.image}")
    
    results = engine.predict_from_image(args.image)
    
    # Save results
    print(f"\nSaving results to: {args.output_dir}")
    engine.save_results(
        results,
        args.output_dir,
        save=args.save_ply
    )
    
    # Print summary
    print("\n" + "="*50)
    print("INFERENCE SUMMARY")
    print("="*50)
    print(f"Input image: {args.image}")
    print(f"W vector shape: {results['w_vector'].shape}")
    print(f"DINO points shape: {results['dino_points'].shape}")
    print(f"Number of Gaussians: {results['splats']['means'].shape[0]}")
    print(f"Results saved to: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    """
    Usage examples:
    
    # Basic inference
    python scripts/inference_feedforward_no_guidance.py \
        --image path/to/image.jpg \
        --encoder_checkpoint path/to/encoder.pth \
        --decoder_checkpoint path/to/decoder.pth \
        --dino_checkpoint path/to/dino_model.pth
    
    # With novel view rendering
    python scripts/inference_feedforward_no_guidance.py \
        --image path/to/image.jpg \
        --encoder_checkpoint path/to/encoder.pth \
        --decoder_checkpoint path/to/decoder.pth \
        --dino_checkpoint path/to/dino_model.pth \
        --render_views \
        --output_dir custom_output
    
    # CPU inference
    python scripts/inference_feedforward_no_guidance.py \
        --image path/to/image.jpg \
        --encoder_checkpoint path/to/encoder.pth \
        --decoder_checkpoint path/to/decoder.pth \
        --dino_checkpoint path/to/dino_model.pth \
        --device cpu
    """
    main()