import os
import json
import numpy as np
import torch
from scene.cameras import MiniCam
from utils.graphics_utils import focal2fov, getProjectionMatrix
from utils.graphics_utils import getWorld2View2


def make_camera(
    R: np.ndarray,
    T: np.ndarray,
    FoVx: float,
    FoVy: float,
    width: int,
    height: int,
    znear: float = 0.01,
    zfar: float = 100.0,
    device: str = "cuda",
) -> MiniCam:
    """
    Build a MiniCam from pose and intrinsics.

    MiniCam is the lightweight camera used for rendering — it holds only the
    precomputed view/projection matrices that the rasterizer needs.

    Args:
        R: 3x3 rotation matrix (world-to-camera, transposed convention)
        T: 3-element translation vector
        FoVx: Horizontal field of view in radians
        FoVy: Vertical field of view in radians
        width: Image width in pixels
        height: Image height in pixels
        znear: Near clipping plane (default: 0.01)
        zfar: Far clipping plane (default: 100.0)
        device: Torch device for the matrices (default: "cuda")

    Returns:
        MiniCam instance ready for use with GaussianSplatRenderer
    """
    world_view_transform = (
        torch.tensor(getWorld2View2(R, T), dtype=torch.float32)
        .transpose(0, 1)
        .to(device)
    )
    projection_matrix = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy)
        .transpose(0, 1)
        .to(device)
    )
    full_proj_transform = (
        world_view_transform.unsqueeze(0)
        .bmm(projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    return MiniCam(
        width=width,
        height=height,
        fovy=FoVy,
        fovx=FoVx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )


def generate_camera_rotation_matrix(camera_to_object, object_vertical_downward):
    camera_to_object = camera_to_object / np.linalg.norm(
        camera_to_object
    )  # last column
    # the second column of rotation matrix is pointing toward the downward vertical direction
    camera_y = (
        object_vertical_downward
        - np.dot(object_vertical_downward, camera_to_object) * camera_to_object
    )
    camera_y = camera_y / np.linalg.norm(camera_y)  # second column
    first_column = np.cross(camera_y, camera_to_object)
    R = np.column_stack((first_column, camera_y, camera_to_object))
    return R


# supply vertical vector in world space
def generate_local_coord(vertical_vector):
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
    horizontal_1 = np.array([1, 1, 1])
    if np.abs(np.dot(horizontal_1, vertical_vector)) < 0.01:
        horizontal_1 = np.array([0.72, 0.37, -0.67])
    # gram schimit
    horizontal_1 = (
        horizontal_1 - np.dot(horizontal_1, vertical_vector) * vertical_vector
    )
    horizontal_1 = horizontal_1 / np.linalg.norm(horizontal_1)
    horizontal_2 = np.cross(horizontal_1, vertical_vector)

    return vertical_vector, horizontal_1, horizontal_2


# scalar (in degrees), scalar (in degrees), scalar, vec3, mat33 = [horizontal_1; horizontal_2; vertical];  -> vec3
def get_point_on_sphere(azimuth, elevation, radius, center, observant_coordinates):
    canonical_coordinates = (
        np.array(
            [
                np.cos(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(elevation / 180.0 * np.pi),
            ]
        )
        * radius
    )

    return center + observant_coordinates @ canonical_coordinates


def get_camera_position_and_rotation(
    azimuth, elevation, radius, view_center, observant_coordinates
):
    # get camera position
    position = get_point_on_sphere(
        azimuth, elevation, radius, view_center, observant_coordinates
    )
    # get rotation matrix
    R = generate_camera_rotation_matrix(
        view_center - position, -observant_coordinates[:, 2]
    )
    return position, R


def get_current_radius_azimuth_and_elevation(
    camera_position, view_center, observesant_coordinates
):
    center2camera = -view_center + camera_position
    radius = np.linalg.norm(center2camera)
    dot_product = np.dot(center2camera, observesant_coordinates[:, 2])
    cosine = dot_product / (
        np.linalg.norm(center2camera) * np.linalg.norm(observesant_coordinates[:, 2])
    )
    elevation = np.rad2deg(np.pi / 2.0 - np.arccos(cosine))
    proj_onto_hori = center2camera - dot_product * observesant_coordinates[:, 2]
    dot_product2 = np.dot(proj_onto_hori, observesant_coordinates[:, 0])
    cosine2 = dot_product2 / (
        np.linalg.norm(proj_onto_hori) * np.linalg.norm(observesant_coordinates[:, 0])
    )

    if np.dot(proj_onto_hori, observesant_coordinates[:, 1]) > 0:
        azimuth = np.rad2deg(np.arccos(cosine2))
    else:
        azimuth = -np.rad2deg(np.arccos(cosine2))
    return radius, azimuth, elevation


def get_camera_view(
    model_path,
    default_camera_index=0,
    center_view_world_space=None,
    observant_coordinates=None,
    show_hint=False,
    init_azimuthm=None,
    init_elevation=None,
    init_radius=None,
    move_camera=False,
    current_frame=0,
    delta_a=0,
    delta_e=0,
    delta_r=0,
):
    """Load one of the default cameras for the scene."""
    cam_path = os.path.join(model_path, "cameras.json")
    with open(cam_path) as f:
        data = json.load(f)

        if show_hint:
            if default_camera_index < 0:
                default_camera_index = 0
            r, a, e = get_current_radius_azimuth_and_elevation(
                data[default_camera_index]["position"],
                center_view_world_space,
                observant_coordinates,
            )
            print("Default camera ", default_camera_index, " has")
            print("azimuth:    ", a)
            print("elevation:  ", e)
            print("radius:     ", r)
            print("Now exit program and set your own input!")
            exit()

        if default_camera_index > -1:
            raw_camera = data[default_camera_index]

        else:
            raw_camera = data[0]  # get data to be modified

            assert init_azimuthm is not None
            assert init_elevation is not None
            assert init_radius is not None

            if move_camera:
                assert delta_a is not None
                assert delta_e is not None
                assert delta_r is not None
                position, R = get_camera_position_and_rotation(
                    init_azimuthm + current_frame * delta_a,
                    init_elevation + current_frame * delta_e,
                    init_radius + current_frame * delta_r,
                    center_view_world_space,
                    observant_coordinates,
                )
            else:
                position, R = get_camera_position_and_rotation(
                    init_azimuthm,
                    init_elevation,
                    init_radius,
                    center_view_world_space,
                    observant_coordinates,
                )
            raw_camera["rotation"] = R.tolist()
            raw_camera["position"] = position.tolist()

        tmp = np.zeros((4, 4))
        tmp[:3, :3] = raw_camera["rotation"]
        tmp[:3, 3] = raw_camera["position"]
        tmp[3, 3] = 1
        C2W = np.linalg.inv(tmp)
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]

        width = raw_camera["width"]
        height = raw_camera["height"]
        fovx = focal2fov(raw_camera["fx"], width)
        fovy = focal2fov(raw_camera["fy"], height)

        return make_camera(R, T, fovx, fovy, width, height)


def create_rotating_cameras(
    center: np.ndarray,
    radius: float,
    num_frames: int = 120,
    elevation: float = 20.0,
    width: int = 800,
    height: int = 600,
    fov: float = 50.0,
) -> list:
    """
    Create cameras on a circular orbit around a center point.

    The orbit uses a Z-up world convention. The camera completes one full
    360-degree rotation over num_frames frames.

    Args:
        center: World-space point to orbit around, shape (3,)
        radius: Distance from the center to the camera
        num_frames: Number of frames (cameras) in the orbit (default: 120)
        elevation: Camera elevation angle in degrees (default: 20.0)
        width: Image width in pixels (default: 800)
        height: Image height in pixels (default: 600)
        fov: Horizontal field of view in degrees (default: 50.0)

    Returns:
        List of MiniCam objects, one per frame

    Example:
        >>> center = splats.positions.mean(dim=0).cpu().numpy()
        >>> cameras = create_rotating_cameras(center, radius=2.0, num_frames=120)
    """
    vertical, horizontal_1, horizontal_2 = generate_local_coord(np.array([0.0, 0.0, 1.0]))
    observant_coordinates = np.row_stack([horizontal_1, horizontal_2, vertical])

    focal = width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
    fovx = focal2fov(focal, width)
    fovy = focal2fov(focal, height)

    cameras = []
    for frame in range(num_frames):
        azimuth = (frame / num_frames) * 360.0
        position, R_cam = get_camera_position_and_rotation(
            azimuth, elevation, radius, center, observant_coordinates
        )

        tmp = np.zeros((4, 4))
        tmp[:3, :3] = R_cam
        tmp[:3, 3] = position
        tmp[3, 3] = 1.0
        C2W = np.linalg.inv(tmp)

        cameras.append(make_camera(C2W[:3, :3].T, C2W[:3, 3], fovx, fovy, width, height))

    return cameras


def create_look_at_camera(
    eye,
    center,
    up,
    fov_deg: float = 60.0,
    width: int = 800,
    height: int = 600,
    device: str = "cuda",
) -> MiniCam:
    """
    Create a camera from a position, look-at point, and up vector.

    Uses the COLMAP/3DGS camera convention:
        +X = right,  +Y = down (image row direction),  +Z = forward

    Args:
        eye: Camera position in world space, shape (3,)
        center: World-space point the camera looks at
        up: World up vector (e.g. [0, 0, 1] for Z-up, [0, 1, 0] for Y-up)
        fov_deg: Horizontal field of view in **degrees** (default: 60.0)
        width: Image width in pixels (default: 800)
        height: Image height in pixels (default: 600)

    Returns:
        MiniCam instance ready for use with GaussianSplatRenderer

    Example:
        >>> cam = create_look_at_camera(
        ...     eye=[0.2, 0.2, 1.0],
        ...     center=[0.0, 0.0, 0.0],
        ...     up=[0.0, 0.0, 1.0],
        ...     fov_deg=60.0,
        ... )
        >>> image = renderer.render(cam, splats)
    """
    eye = np.array(eye, dtype=np.float64)
    center = np.array(center, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    # Camera +Z: points from eye toward center
    forward = center - eye
    forward /= np.linalg.norm(forward)

    # Camera +Y: world-down projected onto the plane perpendicular to forward.
    # (3DGS camera Y = image row direction = downward in world.)
    world_down = -up / np.linalg.norm(up)
    cam_y = world_down - np.dot(world_down, forward) * forward
    cam_y /= np.linalg.norm(cam_y)

    # Camera +X: cross(Y, Z) for a right-handed system
    right = np.cross(cam_y, forward)
    right /= np.linalg.norm(right)

    # Build C2W: columns are camera axes expressed in world space
    C2W = np.eye(4)
    C2W[:3, 0] = right    # camera +X
    C2W[:3, 1] = cam_y   # camera +Y (down)
    C2W[:3, 2] = forward  # camera +Z
    C2W[:3, 3] = eye      # camera position

    W2C = np.linalg.inv(C2W)

    # Convert FoV from degrees to radians; adjust vertical FoV for aspect ratio
    focal = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    fovx = focal2fov(focal, width)
    fovy = focal2fov(focal, height)

    # make_camera expects R_c2w = W2C[:3,:3].T and T_w2c = W2C[:3, 3]
    return make_camera(W2C[:3, :3].T, W2C[:3, 3], fovx, fovy, width, height, device=device)


def camera_from_colmap(
    sparse_path: str,
    image_index: int,
    transform: np.ndarray = None,
) -> MiniCam:
    """
    Load a camera from a COLMAP sparse reconstruction.

    Reads intrinsics and extrinsics from COLMAP binary files. Optionally
    applies a 4x4 SE(3) transform to the camera pose (e.g. a scene
    normalization matrix from PhysGaussian training).

    Args:
        sparse_path: Path to the COLMAP sparse directory containing
                     cameras.bin and images.bin
        image_index: Zero-based index into the sorted list of image IDs
        transform: Optional 4x4 numpy array applied to the camera-to-world
                   matrix. Rotation is re-normalized after the transform.
                   (default: None)

    Returns:
        MiniCam object

    Example:
        >>> cam = camera_from_colmap("model/sparse/0", image_index=5)
        >>> cam = camera_from_colmap("model/sparse/0", image_index=5, transform=T)
    """
    from scene.colmap_loader import (
        read_intrinsics_binary,
        read_extrinsics_binary,
        qvec2rotmat,
    )

    cameras_dict = read_intrinsics_binary(f"{sparse_path}/cameras.bin")
    images_dict = read_extrinsics_binary(f"{sparse_path}/images.bin")
    image_ids = sorted(images_dict.keys())

    img_id = image_ids[image_index]
    img = images_dict[img_id]
    cam_info = cameras_dict[img.camera_id]

    # Build C2W for this camera
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(img.qvec)
    w2c[:3, 3] = img.tvec
    c2w = np.linalg.inv(w2c)

    if transform is not None:
        c2w = transform @ c2w
        # Re-normalize rotation (transform may introduce scale)
        scaling = np.linalg.norm(c2w[0, :3])
        c2w[:3, :3] /= scaling

    w2c_final = np.linalg.inv(c2w)
    R = w2c_final[:3, :3].T
    T = w2c_final[:3, 3]

    # Intrinsics
    if cam_info.model == "PINHOLE":
        fx, fy = cam_info.params[0], cam_info.params[1]
    else:  # SIMPLE_PINHOLE
        fx = fy = cam_info.params[0]

    return make_camera(R, T, focal2fov(fx, cam_info.width), focal2fov(fy, cam_info.height), cam_info.width, cam_info.height), cam_info
