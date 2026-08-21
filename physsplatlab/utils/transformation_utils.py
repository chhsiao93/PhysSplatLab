import torch
import torch.nn.functional as F
import numpy as np
from .camera_view_utils import *


def transform2origin(position_tensor, scale=None, boundary=None):
    device = position_tensor.device
    if boundary is None:
        min_pos = torch.min(position_tensor, 0)[0]
        max_pos = torch.max(position_tensor, 0)[0]
    else:
        # boundary = [min_x, max_x, min_y, max_y, min_z, max_z]
        min_pos = torch.tensor(boundary[::2], device=device, dtype=torch.float32)
        max_pos = torch.tensor(boundary[1::2], device=device, dtype=torch.float32)

    max_diff = torch.max(max_pos - min_pos)
    original_mean_pos = (min_pos + max_pos) / 2.0
    if scale is None:
        scale = torch.tensor(1.0, device=device) / max_diff
    else:
        scale = torch.tensor(scale, device=device)
    new_position_tensor = (position_tensor - original_mean_pos) * scale

    return new_position_tensor, scale, original_mean_pos


def undotransform2origin(position_tensor, scale, original_mean_pos):
    return original_mean_pos + position_tensor / scale


def generate_rotation_matrix(degree, axis):
    cos_theta = torch.cos(degree / 180.0 * 3.1415926)
    sin_theta = torch.sin(degree / 180.0 * 3.1415926)
    if axis == 0:
        rotation_matrix = torch.tensor(
            [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]]
        )
    elif axis == 1:
        rotation_matrix = torch.tensor(
            [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
        )
    elif axis == 2:
        rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
        )
    else:
        raise ValueError("Invalid axis selection")
    return rotation_matrix.float()


def generate_rotation_matrices(degrees, axises):
    assert len(degrees) == len(axises)

    matrices = []

    for i in range(len(degrees)):
        matrices.append(generate_rotation_matrix(degrees[i], axises[i]))

    return matrices


def apply_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix.T)
    return rotated


def apply_cov_rotation(cov_tensor, rotation_matrix):
    rotated = torch.matmul(cov_tensor, rotation_matrix.T)
    rotated = torch.matmul(rotation_matrix, rotated)
    return rotated


def get_mat_from_upper(upper_mat):
    upper_mat = upper_mat.reshape(-1, 6)
    mat = torch.zeros((upper_mat.shape[0], 9), device=upper_mat.device)
    mat[:, :3] = upper_mat[:, :3]
    mat[:, 3] = upper_mat[:, 1]
    mat[:, 4] = upper_mat[:, 3]
    mat[:, 5] = upper_mat[:, 4]
    mat[:, 6] = upper_mat[:, 2]
    mat[:, 7] = upper_mat[:, 4]
    mat[:, 8] = upper_mat[:, 5]

    return mat.view(-1, 3, 3)


def get_uppder_from_mat(mat):
    mat = mat.view(-1, 9)
    upper_mat = torch.zeros((mat.shape[0], 6), device=mat.device)
    upper_mat[:, :3] = mat[:, :3]
    upper_mat[:, 3] = mat[:, 4]
    upper_mat[:, 4] = mat[:, 5]
    upper_mat[:, 5] = mat[:, 8]

    return upper_mat


def apply_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        position_tensor = apply_rotation(position_tensor, rotation_matrices[i])
    return position_tensor


def apply_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        cov_tensor = apply_cov_rotation(cov_tensor, rotation_matrices[i])
    return get_uppder_from_mat(cov_tensor)


def shift2center111(position_tensor):
    return position_tensor + torch.tensor([1.0, 1.0, 1.0], device=position_tensor.device)


def undoshift2center111(position_tensor):
    return position_tensor - torch.tensor([1.0, 1.0, 1.0], device=position_tensor.device)


def shift2center555(position_tensor):
    return position_tensor + torch.tensor([0.5, 0.5, 0.5], device=position_tensor.device)


def undoshift2center555(position_tensor):
    return position_tensor - torch.tensor([0.5, 0.5, 0.5], device=position_tensor.device)


def apply_inverse_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix)
    return rotated


def apply_inverse_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        position_tensor = apply_inverse_rotation(position_tensor, R)
    return position_tensor


def apply_inverse_cov_rotations(upper_cov_tensor, rotation_matrices):
    cov_tensor = get_mat_from_upper(upper_cov_tensor)
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        cov_tensor = apply_cov_rotation(cov_tensor, R.T)
    return get_uppder_from_mat(cov_tensor)


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix/matrices to unit quaternions [w, x, y, z].

    Args:
        R: (N, 3, 3) or (1, 3, 3) rotation matrices

    Returns:
        (N, 4) unit quaternions in [w, x, y, z] order
    """
    if R.ndim == 2:
        R = R.unsqueeze(0)

    N = R.shape[0]
    # Shepperd's method: compute all 4 cases in parallel, select by largest diagonal
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # (N,)

    # Case 1: trace > 0
    s1 = 2.0 * torch.sqrt((trace + 1.0).clamp(min=1e-10))
    w1 = s1 / 4.0
    x1 = (R[:, 2, 1] - R[:, 1, 2]) / s1
    y1 = (R[:, 0, 2] - R[:, 2, 0]) / s1
    z1 = (R[:, 1, 0] - R[:, 0, 1]) / s1

    # Case 2: R[0,0] is largest diagonal
    s2 = 2.0 * torch.sqrt((1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp(min=1e-10))
    w2 = (R[:, 2, 1] - R[:, 1, 2]) / s2
    x2 = s2 / 4.0
    y2 = (R[:, 0, 1] + R[:, 1, 0]) / s2
    z2 = (R[:, 0, 2] + R[:, 2, 0]) / s2

    # Case 3: R[1,1] is largest diagonal
    s3 = 2.0 * torch.sqrt((1.0 + R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2]).clamp(min=1e-10))
    w3 = (R[:, 0, 2] - R[:, 2, 0]) / s3
    x3 = (R[:, 0, 1] + R[:, 1, 0]) / s3
    y3 = s3 / 4.0
    z3 = (R[:, 1, 2] + R[:, 2, 1]) / s3

    # Case 4: R[2,2] is largest diagonal
    s4 = 2.0 * torch.sqrt((1.0 + R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1]).clamp(min=1e-10))
    w4 = (R[:, 1, 0] - R[:, 0, 1]) / s4
    x4 = (R[:, 0, 2] + R[:, 2, 0]) / s4
    y4 = (R[:, 1, 2] + R[:, 2, 1]) / s4
    z4 = s4 / 4.0

    q1 = torch.stack([w1, x1, y1, z1], dim=1)
    q2 = torch.stack([w2, x2, y2, z2], dim=1)
    q3 = torch.stack([w3, x3, y3, z3], dim=1)
    q4 = torch.stack([w4, x4, y4, z4], dim=1)

    # Select per row: case 1 if trace>0, else pick by largest diagonal
    cond1 = (trace > 0).unsqueeze(1).expand(N, 4)
    cond2 = ((R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2])).unsqueeze(1).expand(N, 4)
    cond3 = (R[:, 1, 1] >= R[:, 2, 2]).unsqueeze(1).expand(N, 4)

    q = torch.where(cond1, q1, torch.where(cond2, q2, torch.where(cond3, q3, q4)))
    return F.normalize(q, dim=1)


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two batches of quaternions (Hamilton product).

    Args:
        q1: (N, 4) quaternions [w, x, y, z]
        q2: (N, 4) quaternions [w, x, y, z]

    Returns:
        (N, 4) normalized product quaternions [w, x, y, z]
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return F.normalize(torch.stack([w, x, y, z], dim=1), dim=1)


# input must be (n,3) tensor on cuda
def undo_all_transforms(input, rotation_matrices, scale_origin, original_mean_pos):
    return apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(input), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )
def undo_all_transforms_555(input, rotation_matrices, scale_origin, original_mean_pos):
    return apply_inverse_rotations(
        undotransform2origin(
            undoshift2center555(input), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )

def get_center_view_worldspace_and_observant_coordinate(
    mpm_space_viewpoint_center,
    mpm_space_vertical_upward_axis,
    rotation_matrices,
    scale_origin,
    original_mean_pos,
):
    viewpoint_center_worldspace = undo_all_transforms(
        mpm_space_viewpoint_center, rotation_matrices, scale_origin, original_mean_pos
    )
    mpm_space_up = mpm_space_vertical_upward_axis + mpm_space_viewpoint_center
    worldspace_up = undo_all_transforms(
        mpm_space_up, rotation_matrices, scale_origin, original_mean_pos
    )
    world_space_vertical_axis = worldspace_up - viewpoint_center_worldspace
    viewpoint_center_worldspace = np.squeeze(
        viewpoint_center_worldspace.clone().detach().cpu().numpy(), 0
    )
    vertical, h1, h2 = generate_local_coord(
        np.squeeze(world_space_vertical_axis.clone().detach().cpu().numpy(), 0)
    )
    observant_coordinates = np.column_stack((h1, h2, vertical))

    return viewpoint_center_worldspace, observant_coordinates
