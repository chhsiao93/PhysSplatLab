"""
Render a single image from a COLMAP camera using Gaussian splats.

The T1/T2 transforms below are the scene normalization matrices used during
PhysGaussian training for the sandbox scene. Adjust them for other scenes.
"""

import cv2
import numpy as np

from physsplatlab import GaussianSplatManager, GaussianSplatRenderer
from physsplatlab.utils.camera_view_utils import camera_from_colmap


# Scene-specific normalization matrices (from PhysGaussian training)
T1 = np.array([
    [ 0.23273142, -0.05639757, -0.02462004,  0.29783025],
    [ 0.05639757,  0.15694597,  0.17360305, -0.31232968],
    [-0.02462004, -0.17360305,  0.16494417,  0.03522858],
    [ 0.,          0.,          0.,          1.        ],
])

T2 = np.array([
    [0.1625918,  0.06203924, -0.9847411,  -0.01924499],
    [0.90513744, 0.38792931,  0.1738881,   0.08181817],
    [0.39279782, -0.91959882, 0.00692009,  0.38645594],
    [0.0,        0.0,         0.0,         1.0        ],
])

TRANSFORM = T2 @ T1


if __name__ == "__main__":
    ply_path    = "/geoelements/ChengHsi/PhysGaussian/model/sandbox/point_cloud/iteration_3000/point_cloud.ply"
    sparse_path = "/geoelements/ChengHsi/PhysGaussian/model/sandbox/sparse_register"
    sh_degree   = 3
    index       = 243

    cam, cam_info = camera_from_colmap(sparse_path, index, transform=TRANSFORM)
    # print(f"Camera {index}: {cam_info.image_name}")
    print(f"  Center: {cam.camera_center.cpu().numpy()}")

    splats   = GaussianSplatManager.from_ply(ply_path, sh_degree=sh_degree)
    renderer = GaussianSplatRenderer(sh_degree=sh_degree, bg_color="black")

    image = renderer.render(cam, splats)

    out_path = "output.png"
    cv2.imwrite(out_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_path}")
