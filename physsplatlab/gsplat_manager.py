"""
gsplat_manager (Gaussian Splat Manager) - A class structure for managing Gaussian splat data.

This module provides a clean interface for handling Gaussian splat properties
including positions, covariances, opacities, and spherical harmonics (SHs).
"""

import os
import warnings
import torch
import numpy as np
from typing import Optional, Union, Tuple

from .utils.transformation_utils import (
    transform2origin,
    undotransform2origin,
    shift2center555,
    undoshift2center555,
    apply_rotation,
    apply_rotations,
    apply_cov_rotation,
    apply_cov_rotations,
    apply_inverse_rotation,
    apply_inverse_rotations,
    apply_inverse_cov_rotations,
    get_mat_from_upper,
    get_uppder_from_mat,
    matrix_to_quaternion,
    quaternion_multiply,
)


class GaussianSplatManager:
    """
    Manages Gaussian splat data including positions, covariances, opacities, and colors.

    This class provides a clean interface for common operations like masking,
    rotating, translating, and transforming Gaussian splats.

    Attributes:
        positions (torch.Tensor): Splat positions (N, 3)
        covariances (torch.Tensor): Splat covariances (N, 6) or (N, 3, 3)
        opacities (torch.Tensor): Splat opacities (N, 1)
        shs (torch.Tensor): Spherical harmonics coefficients (N, K, 3)
        scales (torch.Tensor | None): Activated scales (N, 3), populated by from_ply
        rotations (torch.Tensor | None): Normalized quaternions (N, 4) [w,x,y,z], populated by from_ply
        device (str): Device where tensors are stored ('cuda' or 'cpu')
        num_splats (int): Number of splats
    """

    def __init__(
        self,
        positions: Union[torch.Tensor, np.ndarray],
        covariances: Union[torch.Tensor, np.ndarray],
        opacities: Union[torch.Tensor, np.ndarray],
        shs: Union[torch.Tensor, np.ndarray],
        device: str = "cuda",
        scales: Optional[Union[torch.Tensor, np.ndarray]] = None,
        rotations: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        """
        Initialize the GaussianSplatManager with splat data.

        Args:
            positions: Splat positions (N, 3)
            covariances: Splat covariances (N, 6) or (N, 3, 3)
            opacities: Splat opacities (N, 1)
            shs: Spherical harmonics coefficients (N, K, 3)
            device: Device to store tensors on ('cuda' or 'cpu')
            scales: Activated scales (N, 3), optional — populated by from_ply
            rotations: Normalized quaternions (N, 4) [w,x,y,z], optional — populated by from_ply
        """
        self.device = device

        # Convert numpy arrays to torch tensors if needed
        self.positions = self._to_tensor(positions)
        self.covariances = self._to_tensor(covariances)
        self.opacities = self._to_tensor(opacities)
        self.shs = self._to_tensor(shs)
        self.scales = self._to_tensor(scales) if scales is not None else None
        self.rotations = self._to_tensor(rotations) if rotations is not None else None

        # Validate dimensions
        self._validate_dimensions()

    def _to_tensor(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Convert data to torch tensor on the specified device.

        Args:
            data: Input data (numpy array or torch tensor)

        Returns:
            torch.Tensor on the specified device
        """
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            raise TypeError(f"Expected numpy array or torch tensor, got {type(data)}")

    def _validate_dimensions(self):
        """
        Validate that all tensors have compatible dimensions.

        Raises:
            ValueError: If dimensions are incompatible
        """
        n = self.positions.shape[0]

        if self.positions.shape[1] != 3:
            raise ValueError(f"Positions must have shape (N, 3), got {self.positions.shape}")

        if self.covariances.shape[0] != n:
            raise ValueError(f"Covariances first dimension {self.covariances.shape[0]} "
                           f"doesn't match positions {n}")

        if self.opacities.shape[0] != n:
            raise ValueError(f"Opacities first dimension {self.opacities.shape[0]} "
                           f"doesn't match positions {n}")

        if self.shs.shape[0] != n:
            raise ValueError(f"SHs first dimension {self.shs.shape[0]} "
                           f"doesn't match positions {n}")

        if self.scales is not None and self.scales.shape != (n, 3):
            raise ValueError(f"scales must have shape (N, 3), got {self.scales.shape}")

        if self.rotations is not None and self.rotations.shape != (n, 4):
            raise ValueError(f"rotations must have shape (N, 4), got {self.rotations.shape}")

    @property
    def num_splats(self) -> int:
        """Return the number of splats."""
        return self.positions.shape[0]

    def __len__(self) -> int:
        """Return the number of splats."""
        return self.num_splats

    def __repr__(self) -> str:
        """Return string representation of the manager."""
        return (f"GaussianSplatManager(num_splats={self.num_splats}, "
                f"device='{self.device}')")

    @classmethod
    def from_dict(cls, data: dict, device: str = "cuda") -> "GaussianSplatManager":
        """
        Create a GaussianSplatManager from a dictionary.

        Args:
            data: Dictionary containing 'pos', 'cov3D_precomp', 'opacity', 'shs'
            device: Device to store tensors on

        Returns:
            GaussianSplatManager instance
        """
        return cls(
            positions=data["pos"],
            covariances=data["cov3D_precomp"],
            opacities=data["opacity"],
            shs=data["shs"],
            device=device
        )

    def to_dict(self) -> dict:
        """
        Convert the splat data to a dictionary.

        Returns:
            Dictionary with keys 'pos', 'cov3D_precomp', 'opacity', 'shs'
        """
        return {
            "pos": self.positions,
            "cov3D_precomp": self.covariances,
            "opacity": self.opacities,
            "shs": self.shs
        }

    def clone(self) -> "GaussianSplatManager":
        """
        Create a deep copy of the GaussianSplatManager.

        Returns:
            New GaussianSplatManager instance with cloned data
        """
        return GaussianSplatManager(
            positions=self.positions.clone(),
            covariances=self.covariances.clone(),
            opacities=self.opacities.clone(),
            shs=self.shs.clone(),
            device=self.device,
            scales=self.scales.clone() if self.scales is not None else None,
            rotations=self.rotations.clone() if self.rotations is not None else None,
        )

    def to(self, device: str) -> "GaussianSplatManager":
        """
        Move all tensors to a different device.

        Args:
            device: Target device ('cuda' or 'cpu')

        Returns:
            Self for method chaining
        """
        self.device = device
        self.positions = self.positions.to(device)
        self.covariances = self.covariances.to(device)
        self.opacities = self.opacities.to(device)
        self.shs = self.shs.to(device)
        if self.scales is not None:
            self.scales = self.scales.to(device)
        if self.rotations is not None:
            self.rotations = self.rotations.to(device)
        return self

    def to_numpy(self) -> dict:
        """
        Convert all tensors to numpy arrays.

        Returns:
            Dictionary with numpy arrays
        """
        result = {
            "positions": self.positions.detach().cpu().numpy(),
            "covariances": self.covariances.detach().cpu().numpy(),
            "opacities": self.opacities.detach().cpu().numpy(),
            "shs": self.shs.detach().cpu().numpy(),
        }
        if self.scales is not None:
            result["scales"] = self.scales.detach().cpu().numpy()
        if self.rotations is not None:
            result["rotations"] = self.rotations.detach().cpu().numpy()
        return result

    def apply_mask(
        self,
        mask: Union[torch.Tensor, np.ndarray],
        inplace: bool = False
    ) -> "GaussianSplatManager":
        """
        Apply a boolean mask to filter splats.

        Args:
            mask: Boolean mask (N,) where True means keep the splat
            inplace: If True, modify this instance. If False, return a new instance (default: False)

        Returns:
            GaussianSplatManager with filtered splats (new instance if inplace=False, self if inplace=True)

        Example:
            >>> # Filter splats within a bounding box
            >>> mask = (splats.positions[:, 0] > -0.5) & (splats.positions[:, 0] < 0.5)
            >>> filtered_splats = splats.apply_mask(mask, inplace=False)
            >>> print(f"Original: {len(splats)}, Filtered: {len(filtered_splats)}")
        """
        # Convert mask to tensor if needed
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).bool().to(self.device)
        elif isinstance(mask, torch.Tensor):
            mask = mask.bool().to(self.device)
        else:
            raise TypeError(f"Mask must be numpy array or torch tensor, got {type(mask)}")

        # Validate mask shape
        if mask.shape[0] != self.num_splats:
            raise ValueError(f"Mask length {mask.shape[0]} doesn't match number of splats {self.num_splats}")

        # Apply mask to all attributes
        filtered_positions = self.positions[mask]
        filtered_covariances = self.covariances[mask]
        filtered_opacities = self.opacities[mask]
        filtered_shs = self.shs[mask]
        filtered_scales = self.scales[mask] if self.scales is not None else None
        filtered_rotations = self.rotations[mask] if self.rotations is not None else None

        if inplace:
            # Modify this instance
            self.positions = filtered_positions
            self.covariances = filtered_covariances
            self.opacities = filtered_opacities
            self.shs = filtered_shs
            self.scales = filtered_scales
            self.rotations = filtered_rotations
            return self
        else:
            # Return new instance
            return GaussianSplatManager(
                positions=filtered_positions,
                covariances=filtered_covariances,
                opacities=filtered_opacities,
                shs=filtered_shs,
                device=self.device,
                scales=filtered_scales,
                rotations=filtered_rotations,
            )

    def filter_by_bounds(
        self,
        bounds: Union[list, tuple],
        inplace: bool = False
    ) -> "GaussianSplatManager":
        """
        Filter splats within a bounding box.

        Args:
            bounds: Bounding box [x_min, x_max, y_min, y_max, z_min, z_max]
            inplace: If True, modify this instance. If False, return a new instance (default: False)

        Returns:
            GaussianSplatManager with filtered splats

        Example:
            >>> # Extract splats within a box
            >>> bounds = [-0.15, 0.15, -0.2, 0.2, 0.4, 0.8]
            >>> filtered_splats = splats.filter_by_bounds(bounds, inplace=False)
            >>> print(f"Kept {len(filtered_splats)} splats within bounds")
        """
        if len(bounds) != 6:
            raise ValueError(f"Bounds must have 6 elements [x_min, x_max, y_min, y_max, z_min, z_max], got {len(bounds)}")

        # Create mask for bounding box
        mask = torch.ones(self.num_splats, dtype=torch.bool, device=self.device)
        for i in range(3):
            mask = torch.logical_and(mask, self.positions[:, i] > bounds[2 * i])
            mask = torch.logical_and(mask, self.positions[:, i] < bounds[2 * i + 1])

        return self.apply_mask(mask, inplace=inplace)

    def rotate(
        self,
        rotation_matrices: list,
        which: str = "both"
    ) -> "GaussianSplatManager":
        """
        Rotate splat positions and/or covariances in place with a list of rotation matrices.

        Args:
            rotation_matrices: List of 3x3 rotation matrices (from generate_rotation_matrices)
            which: What to rotate - "both", "pos", or "cov" (default: "both")
                   - "both": Rotate both positions and covariances
                   - "pos": Rotate only positions
                   - "cov": Rotate only covariances

        Returns:
            Self for method chaining

        Example:
            >>> from utils.transformation_utils import generate_rotation_matrices
            >>> rotation_matrices = generate_rotation_matrices(
            ...     torch.tensor([45.0, 30.0]),
            ...     [2, 1]  # Z-axis then Y-axis
            ... )
            >>> splats.rotate(rotation_matrices)
        """
        # Validate 'which' parameter
        if which not in ["both", "pos", "cov"]:
            raise ValueError(f"which must be 'both', 'pos', or 'cov', got '{which}'")

        # Apply rotations to positions if needed
        if which in ["both", "pos"]:
            self.positions = apply_rotations(self.positions, rotation_matrices)

        # Apply rotations to covariances if needed
        if which in ["both", "cov"]:
            if self.covariances.shape[1] == 6:
                # Upper triangular format - use apply_cov_rotations
                self.covariances = apply_cov_rotations(self.covariances, rotation_matrices)
            elif self.covariances.shape[1:] == (3, 3):
                # Full 3x3 matrix format - apply each rotation
                cov_mat = self.covariances
                for mat in rotation_matrices:
                    cov_mat = apply_cov_rotation(cov_mat, mat)
                self.covariances = cov_mat
            else:
                raise ValueError(f"Unsupported covariance shape: {self.covariances.shape}")

            # Compose rotation into stored quaternions if present
            if self.rotations is not None:
                R_composite = torch.eye(3, device=self.device)
                for mat in rotation_matrices:
                    R_composite = mat.to(self.device) @ R_composite
                q_R = matrix_to_quaternion(R_composite.unsqueeze(0)).expand(len(self.rotations), -1)
                self.rotations = quaternion_multiply(q_R, self.rotations)

        return self

    def inverse_rotate(
        self,
        rotation_matrices: list,
        which: str = "both"
    ) -> "GaussianSplatManager":
        """
        Apply inverse rotation to splat positions and/or covariances in place.

        This is useful for undoing a previous rotation with a list of rotation matrices.
        The inverse rotations are applied in reverse order automatically.

        Args:
            rotation_matrices: List of 3x3 rotation matrices (the SAME list used in rotate())
            which: What to inverse rotate - "both", "pos", or "cov" (default: "both")
                   - "both": Inverse rotate both positions and covariances
                   - "pos": Inverse rotate only positions
                   - "cov": Inverse rotate only covariances

        Returns:
            Self for method chaining

        Example:
            >>> from utils.transformation_utils import generate_rotation_matrices
            >>> rotation_matrices = generate_rotation_matrices(
            ...     torch.tensor([45.0, 30.0]),
            ...     [2, 1]  # Z-axis then Y-axis
            ... )
            >>> # Forward rotation
            >>> splats.rotate(rotation_matrices)
            >>> # ... do some processing ...
            >>> # Inverse rotation (undo the rotation)
            >>> splats.inverse_rotate(rotation_matrices)
        """
        # Validate 'which' parameter
        if which not in ["both", "pos", "cov"]:
            raise ValueError(f"which must be 'both', 'pos', or 'cov', got '{which}'")

        # Apply inverse rotations to positions if needed
        if which in ["both", "pos"]:
            self.positions = apply_inverse_rotations(self.positions, rotation_matrices)

        # Apply inverse rotations to covariances if needed
        if which in ["both", "cov"]:
            if self.covariances.shape[1] == 6:
                # Upper triangular format - use apply_inverse_cov_rotations
                self.covariances = apply_inverse_cov_rotations(self.covariances, rotation_matrices)
            elif self.covariances.shape[1:] == (3, 3):
                # Full 3x3 matrix format - apply each inverse rotation in reverse order
                cov_mat = self.covariances
                for i in range(len(rotation_matrices)):
                    R = rotation_matrices[len(rotation_matrices) - 1 - i]
                    cov_mat = apply_cov_rotation(cov_mat, R.T)
                self.covariances = cov_mat
            else:
                raise ValueError(f"Unsupported covariance shape: {self.covariances.shape}")

            # Compose inverse rotation into stored quaternions if present
            if self.rotations is not None:
                R_composite = torch.eye(3, device=self.device)
                for i in range(len(rotation_matrices)):
                    R = rotation_matrices[len(rotation_matrices) - 1 - i]
                    R_composite = R.to(self.device).T @ R_composite
                q_R = matrix_to_quaternion(R_composite.unsqueeze(0)).expand(len(self.rotations), -1)
                self.rotations = quaternion_multiply(q_R, self.rotations)

        return self

    def transform_to_mpm_space(
        self,
        target_scale: float = None,
        boundary: Optional[Union[list, tuple]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform splat positions to MPM simulation space.

        This applies: (pos - center) * scale + [0.5, 0.5, 0.5]
        to normalize positions to the MPM grid (typically [0, 1]^3).

        Args:
            target_scale: Target scale for the normalized space (default: None). 
                If None, scale is computed boundary if provided or from gaussian positions.
            boundary: Optional bounding box [x_min, x_max, y_min, y_max, z_min, z_max]
                If None, uses the actual min/max of positions

        Returns:
            Tuple of (scale_factor, original_center) needed for inverse transform

        Example:
            >>> # Transform to MPM space
            >>> scale, center = splats.transform_to_mpm_space(
            ...     target_scale=1.0,
            ...     boundary=[-0.4, 0.4, -0.4, 0.4, -0.4, 0.4]
            ... )
            >>> # Later, transform back
            >>> splats.transform_from_mpm_space(scale, center)
        """
        # Apply transform2origin
        self.positions, scale_factor, original_center = transform2origin(
            self.positions, scale=target_scale, boundary=boundary
        )

        # Apply shift to center at [0.5, 0.5, 0.5]
        self.positions = shift2center555(self.positions)

        return scale_factor, original_center

    def transform_from_mpm_space(
        self,
        scale_factor: torch.Tensor,
        original_center: torch.Tensor
    ) -> "GaussianSplatManager":
        """
        Transform splat positions back from MPM simulation space to original world space.

        This is the inverse of transform_to_mpm_space().

        Args:
            scale_factor: Scale factor returned from transform_to_mpm_space()
            original_center: Original center returned from transform_to_mpm_space()

        Returns:
            Self for method chaining

        Example:
            >>> # Transform to MPM space
            >>> scale, center = splats.transform_to_mpm_space(1.0)
            >>> # ... do MPM simulation ...
            >>> # Transform back to world space
            >>> splats.transform_from_mpm_space(scale, center)
        """
        # Undo shift from [0.5, 0.5, 0.5]
        self.positions = undoshift2center555(self.positions)

        # Undo transform2origin
        self.positions = undotransform2origin(self.positions, scale_factor, original_center)

        return self

    @classmethod
    def merge(
        cls,
        splat_a: "GaussianSplatManager",
        splat_b: "GaussianSplatManager",
        device: str = None
    ) -> "GaussianSplatManager":
        """
        Merge two GaussianSplatManagers by concatenating their splats.

        If the two managers have different SH coefficient counts (K), the one with
        fewer coefficients is zero-padded to match the larger.

        Args:
            splat_a: First GaussianSplatManager
            splat_b: Second GaussianSplatManager
            device: Device for the merged result. Defaults to splat_a's device.

        Returns:
            New GaussianSplatManager containing all splats from both inputs

        Example:
            >>> merged = GaussianSplatManager.merge(splat_a, splat_b)
            >>> print(f"Merged: {len(splat_a)} + {len(splat_b)} = {len(merged)} splats")
        """
        target_device = device or splat_a.device

        positions_a = splat_a.positions.to(target_device)
        positions_b = splat_b.positions.to(target_device)

        covariances_a = splat_a.covariances.to(target_device)
        covariances_b = splat_b.covariances.to(target_device)

        if covariances_a.shape[1:] != covariances_b.shape[1:]:
            raise ValueError(
                f"Covariance shapes are incompatible: {covariances_a.shape} vs {covariances_b.shape}"
            )

        opacities_a = splat_a.opacities.to(target_device)
        opacities_b = splat_b.opacities.to(target_device)

        shs_a = splat_a.shs.to(target_device)
        shs_b = splat_b.shs.to(target_device)

        # Pad SHs along the coefficient axis if degrees differ
        k_a, k_b = shs_a.shape[1], shs_b.shape[1]
        if k_a < k_b:
            pad = torch.zeros(shs_a.shape[0], k_b - k_a, shs_a.shape[2], device=target_device)
            shs_a = torch.cat([shs_a, pad], dim=1)
        elif k_b < k_a:
            pad = torch.zeros(shs_b.shape[0], k_a - k_b, shs_b.shape[2], device=target_device)
            shs_b = torch.cat([shs_b, pad], dim=1)

        # Merge scales/rotations: concatenate if both have them, else drop with a warning
        if splat_a.scales is not None and splat_b.scales is not None:
            scales_merged = torch.cat([splat_a.scales.to(target_device), splat_b.scales.to(target_device)], dim=0)
        else:
            scales_merged = None
            if splat_a.scales is not None or splat_b.scales is not None:
                warnings.warn("One operand is missing scales/rotations; merged result will have scales=None, rotations=None.")

        if splat_a.rotations is not None and splat_b.rotations is not None:
            rotations_merged = torch.cat([splat_a.rotations.to(target_device), splat_b.rotations.to(target_device)], dim=0)
        else:
            rotations_merged = None

        return cls(
            positions=torch.cat([positions_a, positions_b], dim=0),
            covariances=torch.cat([covariances_a, covariances_b], dim=0),
            opacities=torch.cat([opacities_a, opacities_b], dim=0),
            shs=torch.cat([shs_a, shs_b], dim=0),
            device=target_device,
            scales=scales_merged,
            rotations=rotations_merged,
        )

    @classmethod
    def from_ply(
        cls,
        ply_path: str,
        sh_degree: int = 3,
        device: str = "cuda"
    ) -> "GaussianSplatManager":
        """
        Load Gaussian splats from a PLY file.

        This method loads a trained Gaussian Splatting model from a PLY checkpoint file
        and extracts the splat data (positions, covariances, opacities, and SHs).

        Args:
            ply_path: Path to the PLY file (e.g., "model/sandbox/point_cloud/iteration_30000/point_cloud.ply")
            sh_degree: Spherical harmonics degree (default: 3)
            device: Device to store tensors on (default: "cuda")

        Returns:
            GaussianSplatManager instance initialized from the PLY file

        Example:
            >>> splats = GaussianSplatManager.from_ply(
            ...     ply_path="model/sandbox/point_cloud/iteration_30000/point_cloud.ply",
            ...     sh_degree=3
            ... )
            >>> print(f"Loaded {len(splats)} splats")
        """
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY file not found at {ply_path}")

        print(f"Loading Gaussian splats from: {ply_path}")

        from scene.gaussian_model import GaussianModel  # lazy: only needed here

        # Load gaussians using GaussianModel
        gaussians = GaussianModel(sh_degree)
        gaussians.load_ply(ply_path)

        # Extract data from GaussianModel
        positions = gaussians.get_xyz.detach()
        opacities = gaussians.get_opacity.detach()
        shs = gaussians.get_features.detach()
        covariances = gaussians.get_covariance(scaling_modifier=1.0).detach()
        scales = gaussians.get_scaling.detach()    # (N, 3) activated (exp applied)
        rotations = gaussians.get_rotation.detach()  # (N, 4) normalized quaternion [w,x,y,z]

        print(f"Loaded {positions.shape[0]} Gaussian splats")
        print(f"  Positions: {positions.shape}")
        print(f"  Covariances: {covariances.shape}")
        print(f"  Opacities: {opacities.shape}")
        print(f"  SHs: {shs.shape}")
        print(f"  Scales: {scales.shape}")
        print(f"  Rotations: {rotations.shape}")

        # Create instance
        return cls(
            positions=positions,
            covariances=covariances,
            opacities=opacities,
            shs=shs,
            device=device,
            scales=scales,
            rotations=rotations,
        )

    def save_ply(self, path: str) -> None:
        """
        Save Gaussian splats to a standard 3DGS PLY file.

        Requires the splat to have been loaded via from_ply() so that raw
        scales and rotations are available. The output is compatible with
        standard 3DGS viewers and can be reloaded with from_ply().

        Args:
            path: Output path for the PLY file

        Raises:
            ValueError: If scales or rotations are not available (e.g. loaded via from_dict)

        Example:
            >>> splats = GaussianSplatManager.from_ply("input.ply", sh_degree=3)
            >>> splats.save_ply("output.ply")
        """
        if self.scales is None or self.rotations is None:
            raise ValueError(
                "save_ply requires raw scales and rotations. "
                "Load the splat via from_ply() to populate them."
            )

        from plyfile import PlyData, PlyElement

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        xyz = self.positions.detach().cpu().numpy()           # (N, 3)
        normals = np.zeros_like(xyz)                          # (N, 3)

        # DC SH: shs[:, 0:1, :] -> (N, 1, 3) -> transpose -> (N, 3, 1) -> flatten -> (N, 3)
        f_dc = self.shs[:, 0:1, :].detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()

        # Rest SH: shs[:, 1:, :] -> (N, K-1, 3) -> transpose -> (N, 3, K-1) -> flatten -> (N, 3*(K-1))
        f_rest = self.shs[:, 1:, :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        # Inverse sigmoid for opacity
        opacities_clamped = self.opacities.clamp(1e-6, 1.0 - 1e-6)
        opacity_raw = torch.log(opacities_clamped / (1.0 - opacities_clamped)).detach().cpu().numpy()

        # Inverse exp for scales
        scales_raw = torch.log(self.scales.clamp(min=1e-8)).detach().cpu().numpy()

        # Rotations are already normalized quaternions
        rotations_np = self.rotations.detach().cpu().numpy()

        # Build dtype matching standard 3DGS PLY format
        attribute_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        attribute_names += [f'f_dc_{i}' for i in range(3)]
        attribute_names += [f'f_rest_{i}' for i in range(f_rest.shape[1])]
        attribute_names += ['opacity']
        attribute_names += [f'scale_{i}' for i in range(3)]
        attribute_names += [f'rot_{i}' for i in range(4)]
        dtype_full = [(attr, 'f4') for attr in attribute_names]

        N = xyz.shape[0]
        elements = np.empty(N, dtype=dtype_full)
        attributes = np.concatenate(
            [xyz, normals, f_dc, f_rest, opacity_raw, scales_raw, rotations_np],
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        print(f"Saved {N} Gaussian splats to: {path}")

    def recover_scales_rotations(self) -> "GaussianSplatManager":
        """
        Recover raw scales and rotations by eigendecomposing the stored covariances.

        This is the inverse of the covariance computation:
            Σ = R · diag(s²) · Rᵀ  →  eigh(Σ) gives eigenvalues s² and eigenvectors R

        Useful for synthetic splats (e.g. created via from_dict with pre-computed
        covariances) that need to be saved as a standard PLY via save_ply().

        Returns:
            Self for method chaining

        Example:
            >>> flood_splats.recover_scales_rotations()
            >>> merged.recover_scales_rotations()
            >>> merged.save_ply("output.ply")
        """
        cov_mat = get_mat_from_upper(self.covariances)  # (N, 3, 3)
        # Replace NaN/Inf rows (e.g. inactive MPM particles) with a tiny isotropic fallback
        bad = ~torch.isfinite(cov_mat).all(dim=-1).all(dim=-1)  # (N,)
        if bad.any():
            eps = 1e-8
            cov_mat[bad] = torch.eye(3, device=cov_mat.device, dtype=cov_mat.dtype) * eps
        # eigh assumes symmetric matrix and returns eigenvalues in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_mat)  # (N,3), (N,3,3)
        self.scales = eigenvalues.clamp(min=0.0).sqrt()          # (N, 3)
        self.rotations = matrix_to_quaternion(eigenvectors)      # (N, 4) [w,x,y,z]
        return self
