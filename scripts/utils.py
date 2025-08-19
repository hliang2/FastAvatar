"""
Utility Functions for Gaussian Splatting
=========================================
This module provides utility functions for:
- PLY file I/O operations
- Camera and point cloud transformations
- Scene normalization
- Random seed management
"""

import os
import math
import struct
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image
from matplotlib import pyplot as plt
from plyfile import PlyData


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sobel kernels for edge detection (if needed)
SOBEL_KERNEL_X = torch.tensor(
    [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
    dtype=torch.float32
).unsqueeze(0)

SOBEL_KERNEL_Y = torch.tensor(
    [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
    dtype=torch.float32
).unsqueeze(0)


# ============================================================================
# PLY File I/O Operations
# ============================================================================

def load_ply_to_splats(file_path: str) -> nn.ParameterDict:
    """
    Load a PLY file and convert it to a ParameterDict of Gaussian splats.
    
    This function reads a PLY file containing Gaussian splat parameters
    and converts them into a PyTorch ParameterDict suitable for use in
    neural rendering pipelines.
    
    Args:
        file_path: Path to the PLY file containing Gaussian splat data
        
    Returns:
        ParameterDict containing splat parameters:
            - means: 3D positions [N, 3]
            - scales: Scale parameters [N, 3]
            - quats: Rotation quaternions [N, 4]
            - opacities: Opacity values [N]
            - sh0: DC spherical harmonics coefficients [N, 1, 3]
            - shN: Higher-order SH coefficients [N, K, 3]
            
    Raises:
        FileNotFoundError: If the PLY file doesn't exist
        ValueError: If the PLY file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PLY file not found: {file_path}")
    
    # Parse PLY file
    vertices_data, property_names, num_vertices = _parse_ply_file(file_path)
    
    # Convert to numpy array
    vertex_array = np.array(vertices_data)
    
    # Extract component indices
    component_indices = _extract_component_indices(property_names)
    
    # Extract and convert components
    components = _extract_ply_components(vertex_array, component_indices)
    
    # Create ParameterDict
    splats = nn.ParameterDict()
    for key, tensor in components.items():
        splats[key] = nn.Parameter(tensor)
    
    return splats


def _parse_ply_file(file_path: str) -> Tuple[List, List[str], int]:
    """
    Parse PLY file header and data.
    
    Args:
        file_path: Path to PLY file
        
    Returns:
        Tuple of (vertex data, property names, number of vertices)
    """
    vertices_data = []
    property_names = []
    num_vertices = 0
    
    with open(file_path, 'rb') as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        # Extract metadata from header
        for line in header_lines:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('property float'):
                property_name = line.split()[-1]
                property_names.append(property_name)
        
        # Read binary vertex data
        for _ in range(num_vertices):
            values = struct.unpack(
                f"<{len(property_names)}f",
                f.read(len(property_names) * 4)
            )
            vertices_data.append(values)
    
    return vertices_data, property_names, num_vertices


def _extract_component_indices(property_names: List[str]) -> Dict[str, List[int]]:
    """
    Map property names to their component types.
    
    Args:
        property_names: List of property names from PLY header
        
    Returns:
        Dictionary mapping component types to their indices
    """
    component_indices = defaultdict(list)
    
    for i, name in enumerate(property_names):
        if name in ['x', 'y', 'z']:
            component_indices['means'].append(i)
        elif name in ['nx', 'ny', 'nz']:
            component_indices['normals'].append(i)
        elif name.startswith('f_dc_'):
            component_indices['sh0'].append(i)
        elif name.startswith('f_rest_'):
            component_indices['shN'].append(i)
        elif name == 'opacity':
            component_indices['opacities'].append(i)
        elif name.startswith('scale_'):
            component_indices['scales'].append(i)
        elif name.startswith('rot_'):
            component_indices['quats'].append(i)
    
    return component_indices


def _extract_ply_components(
    vertex_array: np.ndarray,
    component_indices: Dict[str, List[int]]
) -> Dict[str, torch.Tensor]:
    """
    Extract and convert PLY components to tensors.
    
    Args:
        vertex_array: Numpy array of vertex data
        component_indices: Dictionary mapping component types to indices
        
    Returns:
        Dictionary of component tensors
    """
    components = {}
    
    # Extract position coordinates
    if component_indices['means']:
        means = vertex_array[:, component_indices['means']]
        components['means'] = torch.tensor(means, dtype=torch.float32)
    
    # Extract scale parameters
    if component_indices['scales']:
        scales = vertex_array[:, component_indices['scales']]
        components['scales'] = torch.tensor(scales, dtype=torch.float32)
    
    # Extract rotation quaternions
    if component_indices['quats']:
        quats = vertex_array[:, component_indices['quats']]
        components['quats'] = torch.tensor(quats, dtype=torch.float32)
    
    # Extract opacity values
    if component_indices['opacities']:
        opacities = vertex_array[:, component_indices['opacities']]
        components['opacities'] = torch.tensor(
            opacities.flatten(),
            dtype=torch.float32
        )
    
    # Process spherical harmonics coefficients
    sh0_indices = component_indices['sh0']
    shN_indices = component_indices['shN']
    
    if sh0_indices and shN_indices:
        # Extract SH data
        sh0_data = vertex_array[:, sh0_indices]
        shN_data = vertex_array[:, shN_indices]
        
        # Assuming RGB (3 channels)
        num_channels = 3
        num_vertices = vertex_array.shape[0]
        
        # Reshape to expected format
        sh0 = sh0_data.reshape(num_vertices, num_channels, -1).transpose(0, 2, 1)
        shN = shN_data.reshape(num_vertices, num_channels, -1).transpose(0, 2, 1)
        
        components['sh0'] = torch.tensor(sh0, dtype=torch.float32)
        components['shN'] = torch.tensor(shN, dtype=torch.float32)
    
    return components


# ============================================================================
# Random Seed Management
# ============================================================================

def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Camera and Scene Transformations
# ============================================================================

def similarity_from_cameras(
    c2w: np.ndarray,
    strict_scaling: bool = False,
    center_method: str = "focus"
) -> np.ndarray:
    """
    Compute a similarity transform to normalize camera poses.
    
    This function computes a 4x4 similarity transformation that:
    1. Aligns the world coordinate system with the camera up axis
    2. Recenters the scene
    3. Rescales based on camera distances
    
    Args:
        c2w: Camera-to-world matrices [N, 4, 4] in OpenCV convention
        strict_scaling: If True, use max distance for scaling; else use median
        center_method: Method for centering ("focus" or "poses")
        
    Returns:
        4x4 similarity transformation matrix
        
    Raises:
        ValueError: If center_method is not recognized
    """
    # Extract translation and rotation components
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]
    
    # Step 1: Align world coordinate system
    R_align = _compute_alignment_rotation(R)
    
    # Apply alignment
    R = R_align @ R
    t = (R_align @ t[..., None])[..., 0]
    
    # Step 2: Recenter the scene
    if center_method == "focus":
        translate = _compute_focus_center(R, t)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method: {center_method}")
    
    # Step 3: Compute scale
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    
    # Build transformation matrix
    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align
    transform[:3, :] *= scale
    
    return transform


def _compute_alignment_rotation(R: np.ndarray) -> np.ndarray:
    """
    Compute rotation to align world up with camera up axes.
    
    Args:
        R: Rotation matrices [N, 3, 3]
        
    Returns:
        3x3 alignment rotation matrix
    """
    # Estimate world up by averaging camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)
    
    up_camspace = np.array([0.0, -1.0, 0.0])
    
    # Compute rotation using Rodrigues' formula
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    
    if c > -1:
        # Standard case
        skew = np.array([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ])
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # Edge case: 180-degree rotation
        R_align = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
    
    return R_align


def _compute_focus_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute scene center using camera focus points.
    
    Args:
        R: Aligned rotation matrices [N, 3, 3]
        t: Aligned translation vectors [N, 3]
        
    Returns:
        3D translation vector for centering
    """
    # Forward vectors
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    
    # Find closest point to origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
    
    return -np.median(nearest, axis=0)


def align_principle_axes(point_cloud: np.ndarray) -> np.ndarray:
    """
    Compute transformation to align point cloud with principal axes.
    
    This function uses PCA to find the principal axes of the point cloud
    and returns a transformation that aligns these axes with the coordinate
    system.
    
    Args:
        point_cloud: Points array [N, 3]
        
    Returns:
        4x4 SE(3) transformation matrix
    """
    # Compute centroid using median for robustness
    centroid = np.median(point_cloud, axis=0)
    
    # Center the point cloud
    centered_points = point_cloud - centroid
    
    # Compute covariance matrix
    covariance_matrix = np.cov(centered_points, rowvar=False)
    
    # Compute eigenvectors (principal axes)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort by eigenvalues (descending)
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    
    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    
    # Create rotation matrix
    rotation_matrix = eigenvectors.T
    
    # Build SE(3) transformation
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid
    
    return transform


def transform_points(
    matrix: np.ndarray,
    points: np.ndarray
) -> np.ndarray:
    """
    Transform 3D points using an SE(3) matrix.
    
    Args:
        matrix: 4x4 SE(3) transformation matrix
        points: Points array [N, 3]
        
    Returns:
        Transformed points [N, 3]
        
    Raises:
        AssertionError: If matrix or points have incorrect shape
    """
    assert matrix.shape == (4, 4), f"Expected 4x4 matrix, got {matrix.shape}"
    assert len(points.shape) == 2 and points.shape[1] == 3, \
        f"Expected Nx3 points array, got {points.shape}"
    
    # Apply rotation and translation
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(
    matrix: np.ndarray,
    camtoworlds: np.ndarray
) -> np.ndarray:
    """
    Transform camera poses using an SE(3) matrix.
    
    Args:
        matrix: 4x4 SE(3) transformation matrix
        camtoworlds: Camera-to-world matrices [N, 4, 4]
        
    Returns:
        Transformed camera-to-world matrices [N, 4, 4]
        
    Raises:
        AssertionError: If inputs have incorrect shapes
    """
    assert matrix.shape == (4, 4), f"Expected 4x4 matrix, got {matrix.shape}"
    assert len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4), \
        f"Expected Nx4x4 array, got {camtoworlds.shape}"
    
    # Apply transformation
    transformed = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    
    # Normalize rotation component to ensure orthogonality
    scaling = np.linalg.norm(transformed[:, 0, :3], axis=1)
    transformed[:, :3, :3] = transformed[:, :3, :3] / scaling[:, None, None]
    
    return transformed


def normalize(
    camtoworlds: np.ndarray,
    points: Optional[np.ndarray] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Normalize camera poses and optionally point cloud.
    
    This function applies a series of transformations to normalize the scene:
    1. Similarity transform based on cameras
    2. Principal axis alignment (if points provided)
    
    Args:
        camtoworlds: Camera-to-world matrices [N, 4, 4]
        points: Optional point cloud [M, 3]
        
    Returns:
        If points is None:
            Tuple of (normalized cameras, transformation matrix)
        If points is provided:
            Tuple of (normalized cameras, normalized points, transformation matrix)
    """
    # Apply similarity transformation
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds)
    
    if points is not None:
        # Transform points with similarity
        points = transform_points(T1, points)
        
        # Align to principal axes
        T2 = align_principle_axes(points)
        camtoworlds = transform_cameras(T2, camtoworlds)
        points = transform_points(T2, points)
        
        # Return with combined transformation
        return camtoworlds, points, T2 @ T1
    else:
        return camtoworlds, T1


# ============================================================================
# Image Processing Utilities (if needed)
# ============================================================================

def apply_sobel_filter(
    image: torch.Tensor,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Sobel edge detection filter to an image.
    
    Args:
        image: Input image tensor [B, C, H, W]
        normalize: Whether to normalize the output
        
    Returns:
        Tuple of (gradient_x, gradient_y)
    """
    # Move kernels to same device as image
    kernel_x = SOBEL_KERNEL_X.to(image.device)
    kernel_y = SOBEL_KERNEL_Y.to(image.device)
    
    # Apply convolution
    grad_x = F.conv2d(image, kernel_x, padding=1)
    grad_y = F.conv2d(image, kernel_y, padding=1)
    
    if normalize:
        # Compute magnitude and normalize
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        grad_x = grad_x / (magnitude + 1e-8)
        grad_y = grad_y / (magnitude + 1e-8)
    
    return grad_x, grad_y

# --- Helper functions ---
def quaternion_to_matrix(q):
    """
    Convert quaternion batch [B, 4] (w, x, y, z) to rotation matrices [B, 3, 3]
    """
    q = F.normalize(q, dim=-1)  # ensure unit norm
    w, x, y, z = q.unbind(dim=-1)  # each of shape [B]

    B = q.shape[0]
    R = torch.empty((B, 3, 3), device=q.device)

    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)

    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)

    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return R

def build_camera_matrix(quat, trans):

    R = quaternion_to_matrix(quat)
    T = torch.eye(4, device=quat.device)
    T[:3,:3] = R
    T[:3, 3] = trans
    return T

def decompose_and_interpolate(cam2world_1, cam2world_2, alpha):
    """
    Decompose matrices into components, interpolate separately, then recombine.
    """
    import torch
    
    # Extract translation (assuming last column of the 4x4 matrix)
    t1 = cam2world_1[:3, 3]
    t2 = cam2world_2[:3, 3]
    
    # Extract rotation matrices (3x3 upper-left part)
    R1 = cam2world_1[:3, :3]
    R2 = cam2world_2[:3, :3]
    
    # Linearly interpolate translation
    t_interp = t1 * (1 - alpha) + t2 * alpha
    
    # Interpolate rotation using spherical linear interpolation (Slerp)
    # We'll convert to quaternions for this
    q1 = _matrix_to_quaternion(R1)
    q2 = _matrix_to_quaternion(R2)
    q_interp = _slerp_quaternion(q1, q2, alpha)
    R_interp = _quaternion_to_matrix(q_interp)
    
    # Construct interpolated camera-to-world matrix
    cam2world_interp = torch.eye(4, device=cam2world_1.device)
    cam2world_interp[:3, :3] = R_interp
    cam2world_interp[:3, 3] = t_interp
    
    return cam2world_interp

def _matrix_to_quaternion(matrix):
    """
    Convert 3x3 rotation matrix to quaternion
    """
    import torch
    
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    
    return torch.stack([w, x, y, z])  # quaternion in w, x, y, z order

def _quaternion_to_matrix(quaternion):
    """
    Convert quaternion to 3x3 rotation matrix
    """
    import torch
    
    w, x, y, z = quaternion
    
    # Normalize quaternion
    norm = torch.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Compute rotation matrix
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz, wx, wy, wz = x*y, x*z, y*z, w*x, w*y, w*z
    
    matrix = torch.zeros((3, 3), device=quaternion.device)
    
    matrix[0, 0] = 1 - 2*(yy + zz)
    matrix[0, 1] = 2*(xy - wz)
    matrix[0, 2] = 2*(xz + wy)
    
    matrix[1, 0] = 2*(xy + wz)
    matrix[1, 1] = 1 - 2*(xx + zz)
    matrix[1, 2] = 2*(yz - wx)
    
    matrix[2, 0] = 2*(xz - wy)
    matrix[2, 1] = 2*(yz + wx)
    matrix[2, 2] = 1 - 2*(xx + yy)
    
    return matrix

def _slerp_quaternion(q1, q2, alpha):
    """
    Spherical linear interpolation between quaternions
    """
    import torch
    
    # Compute the cosine of the angle between the quaternions
    dot = torch.sum(q1 * q2)
    
    # If the dot product is negative, negate one of the quaternions
    # to take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # Clamp dot to valid range
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # Compute the angle
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    
    # Handle the case where quaternions are very close
    if sin_theta_0 < 1e-6:
        return q1 * (1 - alpha) + q2 * alpha
    
    # SLERP formula
    theta = theta_0 * alpha
    sin_theta = torch.sin(theta)
    
    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    
    return s0 * q1 + s1 * q2


def structure_preserving_depth_loss(pred_depth, target_depth, scene_scale=1.0):
    """
    Compute a structure-preserving depth loss between predicted and target depth maps
    that encourages similar structure without enforcing identical depth values.
    
    Args:
        pred_depth: Predicted depth map [B, H, W, 1] or [B, H, W]
        target_depth: Target depth map [B, H, W, 1] or [B, H, W]
        scene_scale: Scale factor for the scene
    """
    # Ensure depth maps have the right shape
    if pred_depth.dim() == 4 and pred_depth.shape[3] == 1:
        pred_depth = pred_depth.squeeze(3)  # [B, H, W]
    if target_depth.dim() == 4 and target_depth.shape[3] == 1:
        target_depth = target_depth.squeeze(3)  # [B, H, W]
   

    # 1. Normalized depth loss
    # Normalize explicitly
#     def normalize_depth(depth_map):
#         mean = depth_map.mean()
#         std = depth_map.std()
#         return (depth_map - mean) / std

#     depth_canonical_norm = normalize_depth(pred_depth)
#     depth_target_norm = normalize_depth(target_depth)

#     # Compute loss explicitly
#     depth_loss = torch.mean((depth_target_norm - depth_canonical_norm)**2)
    
    # 2. Gradient loss
    def gradient(img):
        grad_x = img[:, :, 1:] - img[:, :, :-1]
        grad_y = img[:, 1:, :] - img[:, :-1, :]
        return grad_x, grad_y

    # Compute gradients explicitly
    gx_can, gy_can = gradient(pred_depth)
    gx_tar, gy_tar = gradient(target_depth)

    # Explicit gradient difference loss
    grad_loss = torch.mean((gx_can - gx_tar)**2) + torch.mean((gy_can - gy_tar)**2)
    
    # 4. Combine losses with appropriate weights
    total_loss = (
        1.0 * grad_loss
    ) * scene_scale
    
    return total_loss

def laplacian_sharpness_loss(image):
    gray = 0.2989*image[:,0,:,:] + 0.5870*image[:,1,:,:] + 0.1140*image[:,2,:,:]
    edges = F.conv2d(gray.unsqueeze(1), laplacian_kernel, padding=1)
    sharpness = edges.abs().mean()
    return -sharpness  # explicitly maximizing sharpness

def neighbor_preserving_loss(dist_canonical, target_xyz):
    # canonical_xyz: [N, 3]
    # target_xyz: [N, 3]

#     dist_canonical = torch.cdist(canonical_xyz, canonical_xyz)  # [N, N]
    
    dist_target = torch.cdist(target_xyz, target_xyz)          # [N, N]

    # Normalize explicitly to remove global scale
#     dist_canonical /= dist_canonical.mean()
    dist_target /= dist_target.mean()

    neighbor_loss = torch.mean((dist_canonical - dist_target)**2)
    return neighbor_loss

def local_rigidity_loss(canonical_xyz, target_xyz, neighbor_k=10):
    N = target_xyz.shape[0]

    dist_matrix = torch.cdist(canonical_xyz, canonical_xyz)
    _, neighbors_idx = torch.topk(dist_matrix, neighbor_k+1, largest=False) # includes self

    rigidity_loss = 0
    for i in range(N):
        neighbors = neighbors_idx[i, 1:] # exclude self
        canonical_local = canonical_xyz[neighbors] - canonical_xyz[i]
        target_local = target_xyz[neighbors] - target_xyz[i]

        # rigid transformation explicitly (Procrustes alignment)
        canonical_centered = canonical_local - canonical_local.mean(dim=0)
        target_centered = target_local - target_local.mean(dim=0)

        cov = canonical_centered.T @ target_centered
        U, _, Vt = torch.svd(cov)
        R = Vt @ U.T
        aligned = canonical_centered @ R

        rigidity_loss += torch.mean((aligned - target_centered)**2)

    rigidity_loss /= N
    return rigidity_loss


def smoothness_loss(canonical_scale_diff, canonical_opacity_diff, canonical_means_diff, target_scale, target_opacity, target_means, knn_idx):

    # Target differences (scale & opacity)
    target_scale_diff = target_scale.unsqueeze(1) - target_scale[knn_idx]  # (N, k, scale_dim)
#     target_opacity_diff = target_opacity.unsqueeze(1) - target_opacity[knn_idx]  # (N, k, opacity_dim)
    target_means_diff = target_means.unsqueeze(1) - target_means[knn_idx]  # (N, k, scale_dim)
    
    # Relative difference loss
    scale_loss = ((canonical_scale_diff - target_scale_diff)**2).mean()
#     opacity_loss = ((canonical_opacity_diff - target_opacity_diff)**2).mean()
    means_loss = ((canonical_means_diff - target_means_diff)**2).mean()
    
    loss = scale_loss + means_loss
    return loss

def sobel_edge(img):
    # Convert RGB to grayscale first
    # img shape: [B, 3, H, W]
    gray = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
    gray = gray.unsqueeze(1)  # Add channel dimension back: [B, 1, H, W]
    
    Gx = F.conv2d(gray, Sobel_kernel_x.to(img.device), padding=1)
    Gy = F.conv2d(gray, Sobel_kernel_y.to(img.device), padding=1)
    return torch.sqrt(Gx**2 + Gy**2 + 1e-8)

def edge_loss(rendered, reference):
    edge_rendered = sobel_edge(rendered)
    edge_reference = sobel_edge(reference)
    return F.l1_loss(edge_rendered, edge_reference)

def silhouette_loss(rendered, reference, threshold=0.1, temperature=10.0):
    # Soft thresholding instead of hard
    rendered_sil = torch.sigmoid(temperature * (rendered.mean(dim=1, keepdim=True) - threshold))
    reference_sil = torch.sigmoid(temperature * (reference.mean(dim=1, keepdim=True) - threshold))
    return torch.nn.functional.l1_loss(rendered_sil, reference_sil)

def canny_edge_loss(pred, target, low_threshold=0.1, high_threshold=0.2):
    def get_canny_edges(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny((gray * 255).astype(np.uint8), 
                         int(low_threshold * 255), int(high_threshold * 255))
        return torch.from_numpy(edges / 255.0).to(img.device)
    
    pred_edges = get_canny_edges(pred)
    target_edges = get_canny_edges(target)
    return F.mse_loss(pred_edges, target_edges)

class MultiViewEdgeAlignmentLoss(torch.nn.Module):
    def __init__(self, edge_aligner, target_images, loss_weight=1.0):
        """
        Multi-view edge alignment loss for optimizing global offsets
        
        Args:
            edge_aligner: EdgeBasedAligner instance
            target_images: List of target images (16 views) as numpy arrays
            loss_weight: Weight for this loss component
        """
        super().__init__()
        self.edge_aligner = edge_aligner
        self.loss_weight = loss_weight
        
        # Precompute target edges for all views (these don't change during training)
        self.target_edges = []
        self.valid_views = []
        
        print("Precomputing target edges for all views...")
        for view_idx, target_img in enumerate(target_images):
            try:
                target_edges = self.edge_aligner.detect_edges(target_img)
                if target_edges is not None and np.sum(target_edges) > 100:  # Ensure meaningful edges
                    self.target_edges.append(torch.from_numpy(target_edges / 255.0).float())
                    self.valid_views.append(view_idx)
                    print(f"  View {view_idx}: {np.sum(target_edges > 0)} edge pixels")
                else:
                    print(f"  View {view_idx}: Failed - insufficient edges")
            except Exception as e:
                print(f"  View {view_idx}: Failed - {e}")
        
        print(f"Valid views for edge loss: {len(self.valid_views)}/{len(target_images)}")
        
    def compute_edge_iou_loss(self, rendered_edges, target_edges):
        """
        Compute IoU-based edge loss between two edge tensors
        """
        # Intersection and union
        intersection = torch.sum(rendered_edges * target_edges)
        union = torch.sum(torch.clamp(rendered_edges + target_edges, max=1.0))
        
        if union > 0:
            iou = intersection / union
            return 1.0 - iou  # Loss = 1 - IoU
        else:
            return torch.tensor(1.0, device=rendered_edges.device)
    
    def compute_edge_mse_loss(self, rendered_edges, target_edges):
        """
        Compute MSE loss between edge tensors
        """
        return F.mse_loss(rendered_edges, target_edges)
    
    def forward(self, rendered_images: List[torch.Tensor], 
                use_iou_loss: bool = True) -> torch.Tensor:
        """
        Compute multi-view edge alignment loss
        
        Args:
            rendered_images: List of rendered images [H, W, 3] or [H, W, 4] for each valid view
            use_iou_loss: Use IoU loss (True) or MSE loss (False)
            
        Returns:
            edge_alignment_loss: Scalar loss tensor
        """
        if len(self.valid_views) == 0 or len(rendered_images) == 0:
            return torch.tensor(0.0)
        
        device = rendered_images[0].device
        total_loss = torch.tensor(0.0, device=device)
        valid_count = 0
        
        for i, view_idx in enumerate(self.valid_views):
            if i >= len(rendered_images):
                break
                
            try:
                # Get rendered image for this view
                rendered_img = rendered_images[i]  # [H, W, 3] or [H, W, 4]
                
                # Handle different input formats
                if rendered_img.dim() == 4:  # [B, H, W, C]
                    rendered_img = rendered_img[0]  # Remove batch dim
                if rendered_img.dim() == 3 and rendered_img.shape[2] > 3:  # [H, W, 4]
                    rendered_img = rendered_img[..., :3]  # Take RGB only
                
                # Convert to numpy for edge detection
                rendered_np = rendered_img.detach().cpu().numpy()
                rendered_np = (np.clip(rendered_np, 0, 1) * 255).astype(np.uint8)
                
                # Detect edges in rendered image
                rendered_edges_np = self.edge_aligner.detect_edges(rendered_np)
                
                if rendered_edges_np is None:
                    continue
                
                # Convert to tensor
                rendered_edges = torch.from_numpy(rendered_edges_np / 255.0).float().to(device)
                target_edges = self.target_edges[i].to(device)
                
                # Ensure same size (in case of slight dimension mismatches)
                if rendered_edges.shape != target_edges.shape:
                    min_h = min(rendered_edges.shape[0], target_edges.shape[0])
                    min_w = min(rendered_edges.shape[1], target_edges.shape[1])
                    rendered_edges = rendered_edges[:min_h, :min_w]
                    target_edges = target_edges[:min_h, :min_w]
                
                # Compute edge loss
                if use_iou_loss:
                    view_loss = self.compute_edge_iou_loss(rendered_edges, target_edges)
                else:
                    view_loss = self.compute_edge_mse_loss(rendered_edges, target_edges)
                
                total_loss += view_loss
                valid_count += 1
                
            except Exception as e:
                # Skip problematic views
                print(f"Warning: Edge loss failed for view {view_idx}: {e}")
                continue
        
        # Average over valid views
        if valid_count > 0:
            avg_loss = total_loss / valid_count
            return self.loss_weight * avg_loss
        else:
            return torch.tensor(0.0, device=device)

def mask_alignment_loss(rendered_tensor, mask_tensor, threshold=0.9, loss_weight=0.01, step=0, temperature=10.0):
    """
    Mask alignment loss with debugging capabilities
    
    Args:
        rendered_tensor: [bs, height, width, 3] rendered image
        mask_tensor: [bs, height, width] ground truth mask
        threshold: Threshold for determining rendered foreground
        loss_weight: Weight for this loss component
        debug_save: Whether to save debug images
        step: Current training step (for filename)
    
    Returns:
        mask_loss: Scalar loss tensor
    """
    device = rendered_tensor.device
    
    # Extract first batch item
    rendered_img = rendered_tensor[0]  # [height, width, 3]
    mask_img = mask_tensor[0]          # [height, width]
    
    # For rendered: face = low values, background = high values
    rendered_gray = rendered_img.mean(dim=-1)  # [height, width]
    
    # DIFFERENTIABLE soft thresholding instead of hard thresholding
    rendered_silhouette = torch.sigmoid(temperature * (rendered_gray - threshold))
    
    # For mask: face = high values (white), background = low values (black)
    if mask_img.max() > 1.0:
        mask_img = mask_img / 255.0
    target_mask = (mask_img > 0.5).float()  # This is OK since it's ground truth
    
    # Debug saving
    if step % 500 == 0:
        debug_dir = "debug_masks"
        os.makedirs(debug_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original rendered image
        axes[0].imshow(rendered_img.detach().cpu().numpy())
        axes[0].set_title('Rendered Image\n(white bg, noisy fg)')
        axes[0].axis('off')
        
        # Rendered silhouette
        axes[1].imshow(rendered_silhouette.detach().cpu().numpy(), cmap='gray')
        axes[1].set_title(f'Rendered Silhouette\n(face = dark < {threshold})')
        axes[1].axis('off')
        
        # Target mask
        axes[2].imshow(target_mask.detach().cpu().numpy(), cmap='gray')
        axes[2].set_title('Target Mask\n(face = white > 0.5)')
        axes[2].axis('off')
        
        # Overlay comparison
        overlay = torch.stack([
            rendered_silhouette,  # Red channel: where we detect face
            target_mask,          # Green channel: where face should be
            torch.zeros_like(target_mask)  # Blue channel
        ], dim=-1)
        axes[3].imshow(overlay.detach().cpu().numpy())
        axes[3].set_title('Overlay\n(Red: Rendered face, Green: Target face, Yellow: Match)')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{debug_dir}/mask_debug_step_{step:06d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        intersection = torch.sum(rendered_silhouette * target_mask).item()
        union = torch.sum(rendered_silhouette).item() + torch.sum(target_mask).item() - intersection
        iou = intersection / union if union > 0 else 0
        
        rendered_pixels = torch.sum(rendered_silhouette).item()
        target_pixels = torch.sum(target_mask).item()

    
    # Binary cross entropy loss
    mask_loss = F.mse_loss(rendered_silhouette, target_mask)
    
    return loss_weight * mask_loss