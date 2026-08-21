"""Example 1: Load a Gaussian splat scene and render a single image.

Covers:
  - GaussianSplatManager.from_ply — loading a trained 3DGS PLY
  - create_look_at_camera — building a camera from eye/center/up
  - GaussianSplatRenderer.render — rasterizing splats to an image

Usage:
    python examples/01_load_and_render.py
"""

import cv2

from physsplatlab import GaussianSplatManager, GaussianSplatRenderer
from physsplatlab.utils.camera_view_utils import create_look_at_camera

device = "cuda:0"
ply_path = "output/ply/hicss_flood_input_scene.ply"
out_path = "output/examples/01_snapshot.png"

# Load the scene splats from a trained 3DGS PLY checkpoint.
splats = GaussianSplatManager.from_ply(ply_path, sh_degree=3, device=device)

# A camera is defined by where it sits (eye), what it looks at (center),
# and which world direction is "up". Positions are in the same space as
# the splats, so a good starting point is the scene's own bounding-box center.
scene_center = splats.positions.mean(dim=0).cpu().numpy()
cam = create_look_at_camera(
    eye=scene_center + [0.0, -0.6, 0.4],
    center=scene_center,
    up=[0, 0, 1],
    fov_deg=60.0,
    width=1024,
    height=768,
    device=device,
)

renderer = GaussianSplatRenderer(sh_degree=3, bg_color="white", device=device)
image = renderer.render(cam, splats)  # (H, W, 3) uint8, RGB

cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print(f"Saved: {out_path}")
