"""Example 3: Assign different MPM materials to parts of a splat and simulate.

Loads a single splat (a pickup truck) and splits it into three groups by
position, assigning each group a different physical material:
  - chassis / wheels -> "stationary" (rigid, does not move — acts as a fixed
    obstacle other particles collide against)
  - cargo bed         -> "sand"       (granular, crumbles and piles up)
  - cab                -> "fluid"      (melts and flows off)

Covers:
  - Splitting one GaussianSplatManager into groups with boolean masks
  - GaussianSplatManager.merge to lay groups out contiguously for MPM
  - MPM_Simulator_WARP.set_parameters_for_particles to assign per-group
    materials, and add_surface_collider for simple boundary walls
  - Rendering the sim with a single fixed camera into an MP4

Usage:
    python examples/03_truck_materials.py
"""

import sys
from pathlib import Path

import imageio
import torch
from tqdm import tqdm

from physsplatlab import GaussianSplatManager, GaussianSplatRenderer
from physsplatlab.utils.camera_view_utils import create_look_at_camera

# The MPM solver lives in the warp-mpm submodule, not the physsplatlab package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "warp-mpm"))
from mpm_solver_warp import MPM_Simulator_WARP
import warp as wp

wp.init()
wp.config.verify_cuda = False
device = "cuda:0"

ply_path = "data/truck/truck_trimmed.ply"
out_path = "output/examples/03_truck_materials.mp4"

splats = GaussianSplatManager.from_ply(ply_path, sh_degree=3, device=device)

# ==== Split the truck into three groups by position ====
# The point cloud is real-world scaled (~1.4m long) and already Z-up, so no
# rescaling or rotation is needed before feeding it to MPM.
x, z = splats.positions[:, 0], splats.positions[:, 2]
rigid_mask = z < -0.12                      # chassis + wheels (low z)
sand_mask = (~rigid_mask) & (x < 0.05)      # cargo bed (rear half)
fluid_mask = (~rigid_mask) & (x >= 0.05)    # cab (front half)

rigid_splats = splats.apply_mask(rigid_mask)
sand_splats = splats.apply_mask(sand_mask)
fluid_splats = splats.apply_mask(fluid_mask)
num_rigid = rigid_splats.num_splats
num_sand = sand_splats.num_splats
num_fluid = fluid_splats.num_splats
print(f"rigid: {num_rigid}, sand: {num_sand}, fluid: {num_fluid}")

# Merge back in [rigid | sand | fluid] order so MPM start/end indices below
# line up with contiguous blocks.
splats = GaussianSplatManager.merge(rigid_splats, sand_splats)
splats = GaussianSplatManager.merge(splats, fluid_splats)
num_splats = splats.num_splats

# ==== MPM setup ====
mpm_solver = MPM_Simulator_WARP(10, device=device)
grid_lim = 2.5
n_grid = 125
grid_dx = grid_lim / n_grid
particle_per_cell = 1
particle_volume = grid_dx**3 / particle_per_cell
safe_margin = 3 * grid_dx  # keep particles off the grid boundary (see other example scripts)

# Shift the truck into the MPM domain: centered in x/y, resting just above
# the ground plane in z. `shift`/`scale` map real-world -> MPM space; we
# invert them (pos - shift) / scale whenever we read positions back out for rendering.
scale = 1.0
truck_min = splats.positions.min(dim=0).values
truck_center = splats.positions.mean(dim=0)
shift = torch.tensor([
    grid_lim / 2 - truck_center[0].item() * scale,
    grid_lim / 2 - truck_center[1].item() * scale,
    safe_margin - truck_min[2].item() * scale,
], device=device)

position_tensor = splats.positions * scale + shift
cov_tensor = splats.covariances * (scale ** 2)
volume_tensor = torch.full((num_splats,), particle_volume, device=device)

mpm_solver.load_initial_data_from_torch(
    position_tensor, volume_tensor, tensor_cov=cov_tensor,
    grid_lim=grid_lim, n_grid=n_grid, device=device,
)

# Rigid chassis/wheels: fixed in place, acts as an obstacle for the other two groups.
mpm_solver.set_parameters_for_particles(
    start_idx=0,
    end_idx=num_rigid,
    params_dict={"material": "stationary", "density": 1.0},
    device=device)

# Cargo bed: crumbles into a sand pile under gravity.
mpm_solver.set_parameters_for_particles(
    start_idx=num_rigid,
    end_idx=num_rigid + num_sand,
    params_dict={
        "E": 2000.0,
        "nu": 0.2,
        "material": "sand",
        "density": 1600.0,
        "friction_angle": 45.0,
        "softening": 0.1,
    },
    device=device)

# Cab: melts and flows like a liquid.
mpm_solver.set_parameters_for_particles(
    start_idx=num_rigid + num_sand,
    end_idx=num_splats,
    params_dict={
        "bulk_modulus": 100_000.0,
        "material": "fluid",
        "density": 1000.0,
    },
    device=device)

mpm_solver.set_gravity((0.0, 0.0, -9.81))

# ==== Boundary: ground plane + a loose box around the truck's footprint ====
mpm_solver.add_surface_collider((0.0, 0.0, safe_margin), (0.0, 0.0, 1.0), "slip")
mpm_solver.add_surface_collider((safe_margin, 0.0, 0.0), (1.0, 0.0, 0.0), "slip")
mpm_solver.add_surface_collider((grid_lim - safe_margin, 0.0, 0.0), (-1.0, 0.0, 0.0), "slip")
mpm_solver.add_surface_collider((0.0, safe_margin, 0.0), (0.0, 1.0, 0.0), "slip")
mpm_solver.add_surface_collider((0.0, grid_lim - safe_margin, 0.0), (0.0, -1.0, 0.0), "slip")

# ==== Fixed camera, in the same world space as the original splat positions ====
renderer = GaussianSplatRenderer(sh_degree=3, bg_color="white", device=device)
cam = create_look_at_camera(
    eye=truck_center.cpu().numpy() + [0.0, -1.2, 0.6],
    center=truck_center.cpu().numpy(),
    up=[0, 0, 1],
    fov_deg=60.0,
    width=1024,
    height=768,
    device=device,
)

# ==== Simulation loop ====
n_steps = 2000
dt = 0.0001
fps = 30
visualize_step = 25

with imageio.get_writer(out_path, fps=fps, codec="libx264", pixelformat="yuv420p") as video_writer:
    for k in tqdm(range(1, n_steps + 1), desc="Simulating"):
        mpm_solver.p2g2p(k, dt, device=device)
        if k % visualize_step == 0:
            pos = (mpm_solver.export_particle_x_to_torch() - shift) / scale
            cov = mpm_solver.export_particle_cov_to_torch().view(-1, 6) / (scale ** 2)
            splats.positions = pos[:num_splats]
            splats.covariances = cov[:num_splats]

            image = renderer.render(cam, splats)
            video_writer.append_data(image)

print(f"Saved: {out_path}")
