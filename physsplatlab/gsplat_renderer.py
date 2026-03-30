"""
gsplat_renderer - Renders Gaussian splats from a camera viewpoint.
"""

import math
import numpy as np
import torch
from typing import Union

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from .gsplat_manager import GaussianSplatManager


class GaussianSplatRenderer:
    """
    Renders Gaussian splats from a camera viewpoint.

    Wraps the diff-gaussian-rasterization backend so callers only need to
    supply a camera and a GaussianSplatManager.

    Args:
        sh_degree: Spherical harmonics degree used during rasterization (default: 3)
        bg_color: Background color — "black", "white", an RGB tuple/list of floats,
                  or a (3,) torch.Tensor (default: "black")
        device: Device for the background color tensor (default: "cuda")

    Example:
        >>> renderer = GaussianSplatRenderer(sh_degree=3, bg_color="white")
        >>> splats   = GaussianSplatManager.from_ply("model/point_cloud.ply")
        >>> image    = renderer.render(camera, splats)   # (H, W, 3) uint8
    """

    def __init__(
        self,
        sh_degree: int = 3,
        bg_color: Union[str, tuple, list, torch.Tensor] = "black",
        device: str = "cuda",
    ):
        self.sh_degree = sh_degree
        self.device = device
        self.bg_color = self._parse_bg_color(bg_color)

    def _parse_bg_color(self, bg_color: Union[str, tuple, list, torch.Tensor]) -> torch.Tensor:
        if isinstance(bg_color, str):
            presets = {"black": [0.0, 0.0, 0.0], "white": [1.0, 1.0, 1.0]}
            if bg_color not in presets:
                raise ValueError(f"Unknown bg_color {bg_color!r}. Use 'black', 'white', or an RGB value.")
            return torch.tensor(presets[bg_color], dtype=torch.float32, device=self.device)
        elif isinstance(bg_color, (tuple, list)):
            return torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        elif isinstance(bg_color, torch.Tensor):
            return bg_color.to(dtype=torch.float32, device=self.device)
        else:
            raise TypeError(f"bg_color must be str, tuple, list, or Tensor, got {type(bg_color)}")

    def _build_raster_settings(
        self, camera, scaling_modifier: float = 1.0
    ) -> GaussianRasterizationSettings:
        return GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=math.tan(camera.FoVx * 0.5),
            tanfovy=math.tan(camera.FoVy * 0.5),
            bg=self.bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=self.sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=False
        )

    def render_tensor(
        self,
        camera,
        splats: GaussianSplatManager,
        scaling_modifier: float = 1.0,
    ) -> torch.Tensor:
        """
        Render splats from a camera viewpoint.

        Args:
            camera: GSCamera object
            splats: GaussianSplatManager instance
            scaling_modifier: Uniform scale applied to all splat sizes (default: 1.0)

        Returns:
            Rendered image as (3, H, W) float tensor in [0, 1]
        """
        raster_settings = self._build_raster_settings(camera, scaling_modifier)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        screenspace_points = torch.zeros_like(
            splats.positions, requires_grad=False, device=splats.device
        )

        with torch.cuda.device(splats.device):
            color, radii, invdepths = rasterizer(
                means3D=splats.positions,
                means2D=screenspace_points,
                shs=splats.shs,
                colors_precomp=None,
                opacities=splats.opacities,
                scales=None,
                rotations=None,
                cov3D_precomp=splats.covariances,
            )

        return color.clamp(0.0, 1.0)

    def render(
        self,
        camera,
        splats: GaussianSplatManager,
        scaling_modifier: float = 1.0,
    ) -> np.ndarray:
        """
        Render splats from a camera viewpoint.

        Args:
            camera: GSCamera object
            splats: GaussianSplatManager instance
            scaling_modifier: Uniform scale applied to all splat sizes (default: 1.0)

        Returns:
            Rendered image as (H, W, 3) uint8 numpy array
        """
        tensor = self.render_tensor(camera, splats, scaling_modifier)
        image = tensor.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        return (image * 255).astype(np.uint8)
