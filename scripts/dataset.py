"""
Gaussian Face Dataset Module
============================
This module provides dataset classes for loading and processing face data
for conditional Gaussian Splatting, including COLMAP parsing and face embeddings.
"""

import os
import json
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import assert_never

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm
from pycolmap import SceneManager
from insightface.app import FaceAnalysis
import mediapipe as mp

from utils import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


class Parser:
    """
    COLMAP scene parser for loading camera parameters and 3D points.
    
    This class handles loading and processing COLMAP reconstruction data,
    including camera intrinsics, extrinsics, and 3D point clouds.
    
    Args:
        data_dir: Path to the data directory containing COLMAP output
        factor: Downsampling factor for images (default: 1)
        normalize: Whether to normalize the scene coordinates (default: False)
        test_every: Frequency of test image selection (default: 8)
    """
    
    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        
        # Initialize data structures
        self._init_data_structures()
        
        # Load COLMAP data
        self._load_colmap_data()
        
        # Process camera parameters
        self._process_cameras()
        
        # Load and process images
        self._process_images()
        
        # Apply scene normalization if requested
        if self.normalize:
            self._normalize_scene()
        
        # Compute scene scale
        self._compute_scene_scale()
    
    def _init_data_structures(self):
        """Initialize empty data structures for storing parsed data."""
        self.image_names = []
        self.image_paths = []
        self.camtoworlds = None
        self.camera_ids = []
        self.Ks_dict = {}
        self.params_dict = {}
        self.imsize_dict = {}
        self.mask_dict = {}
        self.mapx_dict = {}
        self.mapy_dict = {}
        self.roi_undist_dict = {}
        self.points = None
        self.points_err = None
        self.points_rgb = None
        self.point_indices = {}
        self.transform = np.eye(4)
        self.bounds = np.array([0.01, 1.0])
        self.scene_scale = 1.0
    
    def _load_colmap_data(self):
        """Load COLMAP reconstruction data from disk."""
        # Find COLMAP directory
        colmap_dir = os.path.join(self.data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(self.data_dir, "sparse")
        
        if not os.path.exists(colmap_dir):
            raise ValueError(f"COLMAP directory {colmap_dir} does not exist.")
        
        # Load COLMAP scene
        self.manager = SceneManager(colmap_dir)
        self.manager.load_cameras()
        self.manager.load_images()
        self.manager.load_points3D()
        
        if len(self.manager.images) == 0:
            raise ValueError("No images found in COLMAP reconstruction.")
    
    def _process_cameras(self):
        """Process camera parameters from COLMAP data."""
        imdata = self.manager.images
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        
        for k in imdata:
            im = imdata[k]
            
            # Extract world-to-camera transformation
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
            
            # Store camera ID
            camera_id = im.camera_id
            self.camera_ids.append(camera_id)
            
            # Process camera intrinsics
            cam = self.manager.cameras[camera_id]
            K = self._get_intrinsic_matrix(cam)
            K[:2, :] /= self.factor
            self.Ks_dict[camera_id] = K
            
            # Process distortion parameters
            params, camtype = self._get_distortion_params(cam)
            self.params_dict[camera_id] = params
            
            # Store image dimensions
            self.imsize_dict[camera_id] = (
                cam.width // self.factor,
                cam.height // self.factor
            )
            self.mask_dict[camera_id] = None
        
        # Convert to camera-to-world transformations
        w2c_mats = np.stack(w2c_mats, axis=0)
        self.camtoworlds = np.linalg.inv(w2c_mats)
        
        # Store camera type for undistortion
        self.camtype = camtype
    
    def _get_intrinsic_matrix(self, camera) -> np.ndarray:
        """
        Extract camera intrinsic matrix.
        
        Args:
            camera: COLMAP camera object
            
        Returns:
            3x3 intrinsic matrix
        """
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    def _get_distortion_params(self, camera) -> Tuple[np.ndarray, str]:
        """
        Extract distortion parameters based on camera type.
        
        Args:
            camera: COLMAP camera object
            
        Returns:
            Tuple of (distortion parameters, camera type string)
        """
        type_ = camera.camera_type
        
        # Map camera types to distortion parameters
        if type_ in [0, "SIMPLE_PINHOLE"]:
            return np.empty(0, dtype=np.float32), "perspective"
        elif type_ in [1, "PINHOLE"]:
            return np.empty(0, dtype=np.float32), "perspective"
        elif type_ in [2, "SIMPLE_RADIAL"]:
            return np.array([camera.k1, 0.0, 0.0, 0.0], dtype=np.float32), "perspective"
        elif type_ in [3, "RADIAL"]:
            return np.array([camera.k1, camera.k2, 0.0, 0.0], dtype=np.float32), "perspective"
        elif type_ in [4, "OPENCV"]:
            return np.array([camera.k1, camera.k2, camera.p1, camera.p2], dtype=np.float32), "perspective"
        elif type_ in [5, "OPENCV_FISHEYE"]:
            return np.array([camera.k1, camera.k2, camera.k3, camera.k4], dtype=np.float32), "fisheye"
        else:
            raise ValueError(f"Unsupported camera type: {type_}")
    
    def _process_images(self):
        """Process image paths and load image metadata."""
        imdata = self.manager.images
        
        # Get image names from COLMAP
        image_names = [imdata[k].name for k in imdata]
        
        # Sort images by filename for consistency
        inds = np.argsort(image_names)
        self.image_names = [image_names[i] for i in inds]
        self.camtoworlds = self.camtoworlds[inds]
        self.camera_ids = [self.camera_ids[i] for i in inds]
        
        # Load extended metadata if available
        self._load_extended_metadata()
        
        # Load bounds if available (for forward-facing scenes)
        self._load_bounds()
        
        # Setup image paths
        self._setup_image_paths()
        
        # Process 3D points
        self._process_3d_points()
        
        # Verify and adjust image dimensions
        self._verify_image_dimensions()
        
        # Setup undistortion maps
        self._setup_undistortion()
    
    def _load_extended_metadata(self):
        """Load extended metadata from JSON file if available."""
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        
        extconf_file = os.path.join(self.data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))
    
    def _load_bounds(self):
        """Load scene bounds from file if available."""
        posefile = os.path.join(self.data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]
    
    def _setup_image_paths(self):
        """Setup paths to image files."""
        # Determine image directory suffix
        if self.factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{self.factor}"
        else:
            image_dir_suffix = ""
        
        colmap_image_dir = os.path.join(self.data_dir, "images")
        image_dir = os.path.join(self.data_dir, "images" + image_dir_suffix)
        
        # Verify directories exist
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")
        
        # Map COLMAP image names to actual image files
        colmap_files = sorted(_get_relative_paths(colmap_image_dir))
        image_files = sorted(_get_relative_paths(image_dir))
        
        # Handle downsampling for JPG images
        if self.factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir,
                image_dir + "_png",
                factor=self.factor
            )
            image_files = sorted(_get_relative_paths(image_dir))
        
        # Create mapping and build paths
        colmap_to_image = dict(zip(colmap_files, image_files))
        self.image_paths = [
            os.path.join(image_dir, colmap_to_image[f])
            for f in self.image_names
        ]
    
    def _process_3d_points(self):
        """Process 3D points and create image-to-points mapping."""
        # Extract 3D point data
        self.points = self.manager.points3D.astype(np.float32)
        self.points_err = self.manager.point3D_errors.astype(np.float32)
        self.points_rgb = self.manager.point3D_colors.astype(np.uint8)
        
        # Create mapping from image names to point indices
        point_indices = {}
        image_id_to_name = {v: k for k, v in self.manager.name_to_image_id.items()}
        
        for point_id, data in self.manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = self.manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        
        self.point_indices = {
            k: np.array(v).astype(np.int32)
            for k, v in point_indices.items()
        }
    
    def _verify_image_dimensions(self):
        """Verify and adjust image dimensions based on actual image size."""
        # Load first image to check dimensions
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        
        # Compare with COLMAP dimensions
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height = actual_height / colmap_height
        s_width = actual_width / colmap_width
        
        # Adjust intrinsics and dimensions if needed
        for camera_id in self.Ks_dict:
            K = self.Ks_dict[camera_id]
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (
                int(width * s_width),
                int(height * s_height)
            )
    
    def _setup_undistortion(self):
        """Setup undistortion maps for each camera."""
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            
            # Skip if no distortion
            if len(params) == 0:
                continue
            
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            
            if self.camtype == "perspective":
                self._setup_perspective_undistortion(camera_id, K, params, width, height)
            elif self.camtype == "fisheye":
                self._setup_fisheye_undistortion(camera_id, K, params, width, height)
            else:
                assert_never(self.camtype)
    
    def _setup_perspective_undistortion(
        self,
        camera_id: int,
        K: np.ndarray,
        params: np.ndarray,
        width: int,
        height: int
    ):
        """Setup undistortion for perspective cameras."""
        K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
            K, params, (width, height), 0
        )
        
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, params, None, K_undist, (width, height), cv2.CV_32FC1
        )
        
        self.mapx_dict[camera_id] = mapx
        self.mapy_dict[camera_id] = mapy
        self.Ks_dict[camera_id] = K_undist
        self.roi_undist_dict[camera_id] = roi_undist
        self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
    
    def _setup_fisheye_undistortion(
        self,
        camera_id: int,
        K: np.ndarray,
        params: np.ndarray,
        width: int,
        height: int
    ):
        """Setup undistortion for fisheye cameras."""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Create undistortion grid
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            indexing="xy"
        )
        
        x1 = (grid_x - cx) / fx
        y1 = (grid_y - cy) / fy
        theta = np.sqrt(x1**2 + y1**2)
        
        r = (1.0
             + params[0] * theta**2
             + params[1] * theta**4
             + params[2] * theta**6
             + params[3] * theta**8)
        
        mapx = (fx * x1 * r + width // 2).astype(np.float32)
        mapy = (fy * y1 * r + height // 2).astype(np.float32)
        
        # Define valid region mask
        mask = np.logical_and(
            np.logical_and(mapx > 0, mapy > 0),
            np.logical_and(mapx < width - 1, mapy < height - 1)
        )
        
        # Compute ROI
        y_indices, x_indices = np.nonzero(mask)
        y_min, y_max = y_indices.min(), y_indices.max() + 1
        x_min, x_max = x_indices.min(), x_indices.max() + 1
        
        mask = mask[y_min:y_max, x_min:x_max]
        
        # Adjust intrinsics for ROI
        K_undist = K.copy()
        K_undist[0, 2] -= x_min
        K_undist[1, 2] -= y_min
        
        roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
        
        self.mapx_dict[camera_id] = mapx
        self.mapy_dict[camera_id] = mapy
        self.Ks_dict[camera_id] = K_undist
        self.roi_undist_dict[camera_id] = roi_undist
        self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
        self.mask_dict[camera_id] = mask
    
    def _normalize_scene(self):
        """Normalize the scene coordinates to a canonical space."""
        # Apply similarity transformation based on cameras
        T1 = similarity_from_cameras(self.camtoworlds)
        self.camtoworlds = transform_cameras(T1, self.camtoworlds)
        self.points = transform_points(T1, self.points)
        
        # Align to principal axes
        T2 = align_principle_axes(self.points)
        self.camtoworlds = transform_cameras(T2, self.camtoworlds)
        self.points = transform_points(T2, self.points)
        
        # Store combined transformation
        self.transform = T2 @ T1
    
    def _compute_scene_scale(self):
        """Compute the scale of the scene based on camera positions."""
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


def _get_relative_paths(path_dir: str) -> List[str]:
    """
    Recursively get relative paths of files in a directory.
    
    Args:
        path_dir: Root directory to search
        
    Returns:
        List of relative file paths
    """
    paths = []
    
    # Patterns to ignore
    ignore_dirs = {'.ipynb_checkpoints', '__pycache__', '.git', '.DS_Store', 'Thumbs.db'}
    ignore_files = {'.DS_Store', 'Thumbs.db', '.gitkeep'}
    
    for root, dirs, files in os.walk(path_dir):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        # Add valid files
        for file in files:
            if file not in ignore_files and not file.startswith('.'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, path_dir)
                paths.append(rel_path)
    
    return paths


def _resize_image_folder(
    image_dir: str,
    resized_dir: str,
    factor: int
) -> str:
    """
    Resize all images in a folder by a given factor.
    
    Args:
        image_dir: Source image directory
        resized_dir: Destination directory for resized images
        factor: Downsampling factor
        
    Returns:
        Path to resized directory
    """
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)
    
    image_files = _get_relative_paths(image_dir)
    
    for image_file in tqdm(image_files, desc="Resizing images"):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir,
            os.path.splitext(image_file)[0] + ".png"
        )
        
        # Skip if already processed
        if os.path.isfile(resized_path):
            continue
        
        # Load and resize image
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor))
        )
        
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        
        imageio.imwrite(resized_path, resized_image)
    
    return resized_dir


class GaussianFaceDataset(Dataset):
    """
    PyTorch dataset for conditional Gaussian Splatting on face images.
    
    This dataset handles loading of:
    - Face images and their transformations
    - COLMAP reconstruction data
    - Face embeddings from InsightFace
    - Latent W vectors for each identity
    
    Args:
        data_root: Path to the dataset root directory
        w_dim: Dimension of latent W vectors (default: 512)
        image_size: Target image size as (height, width) (default: (256, 256))
        seed: Random seed for reproducibility (default: 42)
    """
    
    def __init__(
        self,
        data_root: str,
        w_dim: int = 512,
        image_size: Tuple[int, int] = (256, 256),
        seed: int = 42,
    ):
        super().__init__()
        
        # Store configuration
        self.data_root = data_root
        self.w_dim = w_dim
        self.image_size = image_size
        
        # Set random seeds for reproducibility
        self._set_random_seeds(seed)
        
        # Initialize face analysis model
        self._init_face_analysis()
        
        # Initialize data structures
        self.ids = []
        self.data_samples = []
        
        # Discover and load data
        print("Discovering dataset...")
        self._discover_data()
        
        # Initialize W vectors for identities
        self.w_vectors, self.w_ids_to_idx = self._initialize_w_vectors()
        
        # Setup image transforms
        self._setup_transforms()
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _init_face_analysis(self):
        """Initialize InsightFace model for face embedding extraction."""
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0)  # Use GPU 0
    
    def _setup_transforms(self):
        """Setup image transformation pipeline."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def _discover_data(self):
        """
        Discover all valid data samples in the dataset.
        
        This method parses the COLMAP reconstruction and extracts
        face embeddings for each image.
        """
        # Parse COLMAP data
        self.parser = Parser(self.data_root)
        
        # Add data samples from parsed data
        self._add_data_samples(self.parser, identity_id=0)
    
    def _add_data_samples(self, parser: Parser, identity_id: int):
        """
        Add data samples from a parsed COLMAP reconstruction.
        
        Args:
            parser: COLMAP parser instance
            identity_id: Identity ID for these samples
        """
        print(f"Processing {len(parser.image_paths)} images...")
        
        for i in tqdm(range(len(parser.image_paths)), desc="Extracting face embeddings"):
            img_path = parser.image_paths[i]
            
            # Load image for face detection
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            # Extract face embeddings
            faces = self.face_app.get(img)
            
            if len(faces) == 0:
                print(f"Warning: No face detected in {img_path}")
                continue
            
            # Use the first detected face
            face = faces[0]
            
            # Create data sample
            sample = {
                "means": parser.points,
                "image_path": img_path,
                "camtoworlds": parser.camtoworlds[i],
                "K": parser.Ks_dict[parser.camera_ids[i]],
                "id": identity_id,
                "embedding": face.embedding,
                "image_name": parser.image_names[i],
                "camera_id": parser.camera_ids[i],
            }
            
            self.data_samples.append(sample)
            
            # Track unique identities
            if identity_id not in self.ids:
                self.ids.append(identity_id)
    
    def _initialize_w_vectors(self) -> Tuple[torch.Tensor, Dict[int, int]]:
        """
        Initialize W vectors for each identity.
        
        Returns:
            Tuple of:
                - W vectors tensor [num_identities, w_dim]
                - Dictionary mapping identity IDs to W vector indices
        """
        # Create mapping from identity ID to index
        sorted_ids = sorted(self.ids)
        w_ids_to_idx = {id_val: idx for idx, id_val in enumerate(sorted_ids)}
        
        # Initialize W vectors with random values
        num_identities = len(self.ids)
        w_vectors = torch.randn(num_identities, self.w_dim)
        
        print(f"Initialized {num_identities} W vectors of dimension {self.w_dim}")
        
        return w_vectors, w_ids_to_idx
    
    def get_w_vector(self, identity_id: int) -> torch.Tensor:
        """
        Get the W vector for a specific identity.
        
        Args:
            identity_id: Identity ID
            
        Returns:
            W vector tensor [w_dim]
        """
        idx = self.w_ids_to_idx[identity_id]
        return self.w_vectors[idx]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - means: 3D point positions
                - K: Camera intrinsic matrix
                - camtoworlds: Camera extrinsic matrix
                - id: Identity ID
                - pixels: Raw image pixels
                - image: Transformed image tensor
                - embedding: Face embedding vector
                - w_vector: Latent W vector for the identity
        """
        sample = self.data_samples[idx]
        
        # Load and transform image
        image = Image.open(sample['image_path']).convert('RGB')
        image_tensor = self.transform(image)
        
        # Load raw pixels
        pixels = imageio.imread(sample["image_path"])[..., :3]
        
        # Get W vector for this identity
        w_vector = self.get_w_vector(sample["id"])
        
        # Prepare output dictionary
        return_sample = {
            "means": torch.from_numpy(sample["means"]).float(),
            "K": torch.from_numpy(sample["K"]).float(),
            "camtoworlds": torch.from_numpy(sample["camtoworlds"]).float(),
            "id": sample["id"],
            "pixels": torch.from_numpy(pixels).float(),
            "image": image_tensor,
            "embedding": torch.from_numpy(sample["embedding"]).float(),
            "w_vector": w_vector,
            "image_name": sample["image_name"],
            "camera_id": sample["camera_id"],
        }
        
        return return_sample
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return {
            "num_samples": len(self.data_samples),
            "num_identities": len(self.ids),
            "w_dim": self.w_dim,
            "image_size": self.image_size,
            "identity_ids": self.ids,
            "samples_per_identity": {
                id_: sum(1 for s in self.data_samples if s["id"] == id_)
                for id_ in self.ids
            }
        }