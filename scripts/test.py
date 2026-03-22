from physsplatlab import GaussianSplatManager, GaussianSplatRenderer
from physsplatlab.utils.camera_view_utils import create_look_at_camera
import cv2

device = "cuda:0"

ply_path = "data/alligator/scene_point_cloud.ply"

renderer = GaussianSplatRenderer(sh_degree=3, bg_color="black", device=device)
cam = create_look_at_camera(
    eye=[0.2, 0.2, 0.5],
    center=[0.0, 0.0, 0],
    up=[0, 0, 1]
)
splats   = GaussianSplatManager.from_ply(ply_path, sh_degree=3, device=device)
print(f"xyz range: {splats.positions.min(dim=0).values.cpu().numpy()} to {splats.positions.max(dim=0).values.cpu().numpy()}")
image = renderer.render(cam, splats)
out_path = "output.png"
cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print(f"Saved: {out_path}")