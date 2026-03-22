import sys
import os

# Add gaussian-splatting to path so its modules (scene, utils, gaussian_renderer)
# are importable anywhere in this package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gaussian-splatting"))

from .gsplat_manager import GaussianSplatManager
from .gsplat_renderer import GaussianSplatRenderer

__all__ = ["GaussianSplatManager", "GaussianSplatRenderer"]
