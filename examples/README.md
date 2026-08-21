# Examples

Runnable scripts that walk through the core `physsplatlab` API, from loading
a splat to running an MPM physics simulation on it. Run them from the repo
root with the project environment active:

```bash
source .venv/bin/activate
python examples/01_load_and_render.py
python examples/02_orbit_video.py
python examples/03_truck_materials.py
```

Each script downloads nothing — the splat PLYs they use are checked into
the repo:

| File | Used by |
| --- | --- |
| `examples/ply/hicss_flood_input_scene.ply` | 01, 02 |
| `examples/ply/truck_trimmed.ply` | 03 |

Outputs are written to `output/examples/`.

## Scripts

**`01_load_and_render.py`** — Load a splat PLY with `GaussianSplatManager.from_ply`,
build a camera with `create_look_at_camera`, and render one image with
`GaussianSplatRenderer`.

**`02_orbit_video.py`** — Build a ring of cameras around a scene with
`create_rotating_cameras` and render them into an orbiting MP4.

**`03_truck_materials.py`** — Load one splat (a truck) and split it into three
groups by position, assigning each a different MPM material — the chassis/wheels
stay `stationary` (rigid), the cargo bed becomes `sand`, and the cab becomes
`fluid`. Simulates with `MPM_Simulator_WARP` and renders from a fixed camera.
Shows how to hand off a `GaussianSplatManager` to the MPM solver and assign
per-particle-range materials.
