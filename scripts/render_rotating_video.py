"""
Render a rotating-orbit video from a Gaussian splat PLY file.
"""

import argparse
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import torch
from tqdm import tqdm

from physsplatlab import GaussianSplatManager, GaussianSplatRenderer
from physsplatlab.utils.camera_view_utils import create_rotating_cameras
from physsplatlab.utils.transformation_utils import generate_rotation_matrices


def main():
    parser = argparse.ArgumentParser(description="Render rotating video from Gaussian splats")
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to the PLY file")
    parser.add_argument("--output", type=str, default="rotating_video.mp4",
                        help="Output video file path")
    parser.add_argument("--num_frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--radius", type=float, default=None,
                        help="Camera distance from center (auto-computed if omitted)")
    parser.add_argument("--elevation", type=float, default=20.0,
                        help="Camera elevation angle in degrees")
    parser.add_argument("--fov", type=float, default=50.0,
                        help="Horizontal field of view in degrees")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_frames", action="store_true",
                        help="Also save individual frames as PNGs")
    parser.add_argument("--frames_dir", type=str, default="rendered_frames",
                        help="Directory for individual frames (used with --save_frames)")
    args = parser.parse_args()

    # --- Load splats ---
    splats = GaussianSplatManager.from_ply(args.ply_path, device=args.device)

    # Optional: apply an initial rotation before rendering
    rotation_matrices = generate_rotation_matrices(torch.tensor([0]), [0])
    # to device
    rotation_matrices = [m.to(args.device) for m in rotation_matrices]
    splats.rotate(rotation_matrices)

    # --- Auto-compute radius ---
    center = splats.positions.mean(dim=0).cpu().numpy()
    if args.radius is None:
        distances = torch.norm(splats.positions - splats.positions.mean(dim=0), dim=1)
        args.radius = distances.max().item() * 2.0
        print(f"Auto-computed radius: {args.radius:.3f}")

    # --- Build cameras and renderer ---
    cameras = create_rotating_cameras(
        center=center,
        radius=args.radius,
        num_frames=args.num_frames,
        elevation=args.elevation,
        width=args.width,
        height=args.height,
        fov=args.fov,
    )

    renderer = GaussianSplatRenderer(
        sh_degree=3,
        bg_color="white" if args.white_background else "black",
        device=args.device,
    )

    # --- Render ---
    if args.save_frames:
        os.makedirs(args.frames_dir, exist_ok=True)

    frames = []
    for i, camera in enumerate(tqdm(cameras, desc="Rendering")):
        frame = renderer.render(camera, splats)
        frames.append(frame)
        if args.save_frames:
            cv2.imwrite(
                os.path.join(args.frames_dir, f"frame_{i:04d}.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )

    # --- Encode video ---
    print(f"Saving video to {args.output}...")

    if args.save_frames:
        frame_dir = args.frames_dir
        cleanup = False
    else:
        frame_dir = tempfile.mkdtemp()
        cleanup = True
        for i, frame in enumerate(tqdm(frames, desc="Writing frames")):
            cv2.imwrite(
                os.path.join(frame_dir, f"frame_{i:04d}.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )

    result = subprocess.run([
        "ffmpeg", "-framerate", str(args.fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", args.output,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ffmpeg failed: {result.stderr}\nFalling back to OpenCV...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    if cleanup:
        shutil.rmtree(frame_dir)

    print(f"Done: {args.output}  ({args.width}x{args.height} @ {args.fps} fps, {args.num_frames} frames)")


if __name__ == "__main__":
    main()
