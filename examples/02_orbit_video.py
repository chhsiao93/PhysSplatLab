"""Example 2: Orbit a camera around a splat scene and render a video.

Covers:
  - create_rotating_cameras — generating a ring of cameras around a center point
  - Rendering a sequence of frames and writing them out as an MP4 with imageio

Usage:
    python examples/02_orbit_video.py
"""

import imageio
from tqdm import tqdm

from physsplatlab import GaussianSplatManager, GaussianSplatRenderer
from physsplatlab.utils.camera_view_utils import create_rotating_cameras

device = "cuda:0"
ply_path = "examples/ply/hicss_flood_input_scene.ply"
out_path = "output/examples/02_orbit.mp4"

splats = GaussianSplatManager.from_ply(ply_path, sh_degree=3, device=device)
renderer = GaussianSplatRenderer(sh_degree=3, bg_color="white", device=device)

# Orbit around the scene's bounding-box center. `elevation` tilts the ring up
# from the horizontal plane; `radius` controls distance (tune to frame the scene).
scene_center = splats.positions.mean(dim=0).cpu().numpy()
scene_extent = (splats.positions.max(dim=0).values - splats.positions.min(dim=0).values).max().item()
cameras = create_rotating_cameras(
    center=scene_center,
    radius=scene_extent * 1.2,
    num_frames=120,
    elevation=25.0,
    width=1024,
    height=768,
    fov=60.0,
)

with imageio.get_writer(out_path, fps=30, codec="libx264", pixelformat="yuv420p") as video_writer:
    for cam in tqdm(cameras, desc="Rendering orbit"):
        image = renderer.render(cam, splats)
        video_writer.append_data(image)

print(f"Saved: {out_path}")
