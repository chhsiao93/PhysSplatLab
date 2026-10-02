"""
COLMAP sparse-model IO and gsplat's world-space normalization.

gsplat's COLMAP parser (examples/datasets/colmap.py, normalize=True) applies a
similarity transform to the scene before training and never saves it, so the
trained splats live in a normalized frame. normalize_world() recomputes that
transform from the same sparse model, mirroring gsplat's
examples/datasets/{normalize.py,colmap.py}. If gsplat changes that logic,
diff those files against this one.

Example:
    >>> c2w = read_images_binary_c2w(f"{sparse_dir}/images.bin")
    >>> points = read_points3d_binary_xyz(f"{sparse_dir}/points3D.bin")
    >>> T = normalize_world(c2w, points)                    # colmap world -> splat
    >>> splats.apply_similarity_transform(np.linalg.inv(T))  # splat -> colmap world
"""

import struct

import numpy as np


def _read(fid, num_bytes, fmt):
    return struct.unpack("<" + fmt, fid.read(num_bytes))


def qvec2rotmat(q):
    """COLMAP quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
    ])


def read_images_binary_c2w(path):
    """images.bin -> (N, 4, 4) camera-to-world matrices, in file order, same
    convention as gsplat's parser (inverse of COLMAP's w2c, no axis remapping)."""
    c2w = []
    with open(path, "rb") as fid:
        num_images = _read(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = _read(fid, 64, "idddddddi")
            qvec, tvec = np.array(props[1:5]), np.array(props[5:8])
            while _read(fid, 1, "c")[0] != b"\x00":  # image name
                pass
            num_points2d = _read(fid, 8, "Q")[0]
            fid.seek(24 * num_points2d, 1)            # (x, y, point3D_id) per keypoint
            w2c = np.eye(4)
            w2c[:3, :3] = qvec2rotmat(qvec)
            w2c[:3, 3] = tvec
            c2w.append(np.linalg.inv(w2c))
    return np.stack(c2w, axis=0)


def read_points3d_binary(path):
    """points3D.bin -> xyz (M, 3), rgb (M, 3) uint8, reprojection error (M,),
    track length (M,) = number of images observing each point."""
    with open(path, "rb") as fid:
        num_points = _read(fid, 8, "Q")[0]
        xyz = np.zeros((num_points, 3))
        rgb = np.zeros((num_points, 3), dtype=np.uint8)
        error = np.zeros(num_points)
        track_len = np.zeros(num_points, dtype=np.int64)
        for i in range(num_points):
            props = _read(fid, 43, "QdddBBBd")       # id, xyz, rgb, error
            xyz[i] = props[1:4]
            rgb[i] = props[4:7]
            error[i] = props[7]
            track_len[i] = _read(fid, 8, "Q")[0]
            fid.seek(8 * track_len[i], 1)             # (image_id, point2D_idx) per observation
    return xyz, rgb, error, track_len


def read_points3d_binary_xyz(path):
    """points3D.bin -> (M, 3) point positions."""
    return read_points3d_binary(path)[0]


def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """Align mean camera up to [0, -1, 0], center on the cameras' focus point
    (or pose median), and scale so the median camera distance is 1."""
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    world_up = np.sum(R * np.array([0, -1.0, 0]), axis=-1).mean(axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) / (1 + c)
    else:
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    if center_method == "focus":
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale
    return transform


def align_principal_axes(points):
    """Rotate so the point cloud's principal axes (largest variance first) map to x, y, z."""
    centroid = np.median(points, axis=0)
    cov = np.cov(points - centroid, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    transform = np.eye(4)
    transform[:3, :3] = eigenvectors.T
    transform[:3, 3] = -eigenvectors.T @ centroid
    return transform


def transform_points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, c2w):
    c2w = np.einsum("nij, ki -> nkj", c2w, matrix)
    scaling = np.linalg.norm(c2w[:, 0, :3], axis=1)
    c2w[:, :3, :3] = c2w[:, :3, :3] / scaling[:, None, None]
    return c2w


def normalize_world(c2w, points):
    """
    Recompute gsplat's normalize_world_space transform (colmap world -> splat).

    Mirrors the "Normalize the world space" block of gsplat's Parser,
    including the upside-down flip. Deterministic, so the result matches what
    gsplat applied at training time as long as c2w / points come from the same
    sparse model passed as --data_dir.

    Args:
        c2w: (N, 4, 4) camera-to-world matrices (read_images_binary_c2w)
        points: (M, 3) sparse points (read_points3d_binary_xyz)

    Returns:
        (4, 4) similarity transform; invert it to map splats back to the
        COLMAP world frame
    """
    T1 = similarity_from_cameras(c2w)
    c2w = transform_cameras(T1, c2w)
    points = transform_points(T1, points)

    T2 = align_principal_axes(points)
    c2w = transform_cameras(T2, c2w)
    points = transform_points(T2, points)

    transform = T2 @ T1

    if np.median(points[:, 2]) > np.mean(points[:, 2]):
        T3 = np.diag([1.0, -1.0, -1.0, 1.0])
        transform = T3 @ transform
    return transform
