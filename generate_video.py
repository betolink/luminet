#!/usr/bin/env python3
"""
Generate video animations of black hole visualizations.

Creates animations by varying black hole parameters across frames:
- Rotation (varying inclination angle)
- Orbital view (azimuthal rotation)
- Zoom (varying outer disk radius)

Usage:
    python generate_video.py --type rotation --frames 60 --output bh_rotation.mp4
    python generate_video.py --type orbit --frames 120 --resolution 1080p
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from luminet.black_hole import BlackHole
from luminet.backends import get_backend
from luminet import black_hole_math as bhmath


def generate_rotation_frames(
    output_dir: Path,
    n_frames: int = 60,
    resolution: tuple = (1920, 1080),
    backend: str = 'taichi',
    hw: str = 'gpu',
    fps: int = 30
):
    """Generate frames showing black hole rotation (varying inclination).
    
    Args:
        output_dir: Directory to save frames
        n_frames: Number of frames to generate
        resolution: (width, height) in pixels
        backend: Computational backend ('scipy', 'taichi', etc.)
        hw: Hardware for taichi ('cpu' or 'gpu')
        fps: Target frames per second
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup backend
    if backend == 'taichi':
        backend_obj = get_backend('taichi', arch=hw)
        bhmath._backend = backend_obj
    else:
        bhmath.set_backend(backend)
    
    # Inclination sweep: 0° to 90° (edge-on to face-on)
    incl_min = 0.1  # Almost face-on (rad)
    incl_max = np.pi / 2 - 0.1  # Almost edge-on (rad)
    inclinations = np.linspace(incl_min, incl_max, n_frames)
    
    # Figure setup
    dpi = 100
    figsize = (resolution[0] / dpi, resolution[1] / dpi)
    
    print(f"Generating {n_frames} frames at {resolution[0]}×{resolution[1]}...")
    print(f"Backend: {backend} ({hw if backend == 'taichi' else 'N/A'})")
    print(f"Inclination range: {np.degrees(incl_min):.1f}° to {np.degrees(incl_max):.1f}°")
    print()
    
    for i, incl in enumerate(tqdm(inclinations, desc="Rendering frames")):
        # Create black hole with current inclination
        bh = BlackHole(
            mass=1.0,
            incl=incl,
            acc=1.0,
            outer_edge=20.0,
            angular_resolution=200,
            radial_resolution=200
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Plot black hole
        bh.plot(ax=ax)
        
        # Add title with current angle
        ax.set_title(f'Black Hole Inclination: {np.degrees(incl):.1f}°', 
                     fontsize=16, color='white')
        
        # Clean layout
        ax.set_aspect('equal')
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        # Save frame
        frame_path = output_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=dpi, facecolor='black', edgecolor='none')
        plt.close(fig)
    
    print(f"\n✅ Generated {n_frames} frames in {output_dir}")
    return output_dir


def generate_orbit_frames(
    output_dir: Path,
    n_frames: int = 120,
    resolution: tuple = (1920, 1080),
    backend: str = 'taichi',
    hw: str = 'gpu',
    fps: int = 30
):
    """Generate frames showing orbital rotation around black hole.
    
    Simulates orbiting around the black hole by rotating the viewing angle.
    Note: True orbital view would require 3D rendering, this approximates
    by varying inclination in a sinusoidal pattern.
    
    Args:
        output_dir: Directory to save frames
        n_frames: Number of frames to generate
        resolution: (width, height) in pixels
        backend: Computational backend
        hw: Hardware for taichi
        fps: Target frames per second
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup backend
    if backend == 'taichi':
        backend_obj = get_backend('taichi', arch=hw)
        bhmath._backend = backend_obj
    else:
        bhmath.set_backend(backend)
    
    # Orbital motion: smooth sinusoidal variation
    base_incl = 1.2  # Base viewing angle (rad)
    incl_variation = 0.3  # Variation amplitude
    phases = np.linspace(0, 2 * np.pi, n_frames)
    inclinations = base_incl + incl_variation * np.sin(phases)
    
    # Figure setup
    dpi = 100
    figsize = (resolution[0] / dpi, resolution[1] / dpi)
    
    print(f"Generating {n_frames} orbital frames at {resolution[0]}×{resolution[1]}...")
    print(f"Backend: {backend} ({hw if backend == 'taichi' else 'N/A'})")
    print()
    
    for i, incl in enumerate(tqdm(inclinations, desc="Rendering frames")):
        bh = BlackHole(
            mass=1.0,
            incl=incl,
            acc=1.0,
            outer_edge=20.0,
            angular_resolution=200,
            radial_resolution=200
        )
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        bh.plot(ax=ax)
        
        # Add orbit progress indicator
        progress = (i / n_frames) * 360
        ax.set_title(f'Orbital Position: {progress:.0f}°', 
                     fontsize=16, color='white')
        
        ax.set_aspect('equal')
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        frame_path = output_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=dpi, facecolor='black', edgecolor='none')
        plt.close(fig)
    
    print(f"\n✅ Generated {n_frames} frames in {output_dir}")
    return output_dir


def generate_zoom_frames(
    output_dir: Path,
    n_frames: int = 60,
    resolution: tuple = (1920, 1080),
    backend: str = 'taichi',
    hw: str = 'gpu',
    fps: int = 30
):
    """Generate frames showing zoom in/out on black hole.
    
    Args:
        output_dir: Directory to save frames
        n_frames: Number of frames to generate
        resolution: (width, height) in pixels
        backend: Computational backend
        hw: Hardware for taichi
        fps: Target frames per second
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup backend
    if backend == 'taichi':
        backend_obj = get_backend('taichi', arch=hw)
        bhmath._backend = backend_obj
    else:
        bhmath.set_backend(backend)
    
    # Zoom: vary outer disk radius
    radius_min = 10.0
    radius_max = 40.0
    # Smooth zoom in and out
    radii = np.concatenate([
        np.linspace(radius_max, radius_min, n_frames // 2),  # Zoom in
        np.linspace(radius_min, radius_max, n_frames // 2),  # Zoom out
    ])
    
    # Figure setup
    dpi = 100
    figsize = (resolution[0] / dpi, resolution[1] / dpi)
    
    print(f"Generating {n_frames} zoom frames at {resolution[0]}×{resolution[1]}...")
    print(f"Backend: {backend} ({hw if backend == 'taichi' else 'N/A'})")
    print()
    
    for i, radius in enumerate(tqdm(radii, desc="Rendering frames")):
        bh = BlackHole(
            mass=1.0,
            incl=1.4,
            acc=1.0,
            outer_edge=radius,
            angular_resolution=200,
            radial_resolution=200
        )
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        bh.plot(ax=ax)
        
        # Maintain consistent axis limits for smooth zoom
        max_extent = radius_max * 1.2
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        
        ax.set_title(f'Disk Radius: {radius:.1f}M', 
                     fontsize=16, color='white')
        
        ax.set_aspect('equal')
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        frame_path = output_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=dpi, facecolor='black', edgecolor='none')
        plt.close(fig)
    
    print(f"\n✅ Generated {n_frames} frames in {output_dir}")
    return output_dir


def create_video_from_frames(frames_dir: Path, output_path: Path, fps: int = 30):
    """Create video from frame images using ffmpeg.
    
    Args:
        frames_dir: Directory containing frame_XXXX.png files
        output_path: Output video path (.mp4)
        fps: Frames per second
    """
    import subprocess
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n⚠️  ffmpeg not found. Please install ffmpeg to create video:")
        print("    sudo apt install ffmpeg  # Ubuntu/Debian")
        print("    brew install ffmpeg      # macOS")
        print()
        print(f"Frames saved in: {frames_dir}")
        print("You can manually create video with:")
        print(f"    ffmpeg -framerate {fps} -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_path}")
        return False
    
    # Create video
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-framerate', str(fps),
        '-i', str(frames_dir / 'frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'slow',
        '-crf', '18',  # High quality
        str(output_path)
    ]
    
    print(f"\nCreating video: {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Video created: {output_path}")
        
        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ Error creating video:")
        print(result.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate black hole animation videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rotation animation (inclination sweep)
  python generate_video.py --type rotation --frames 60 --fps 30
  
  # Orbital animation (smooth orbit)
  python generate_video.py --type orbit --frames 120 --fps 30
  
  # Zoom animation
  python generate_video.py --type zoom --frames 60
  
  # High quality 1080p with GPU
  python generate_video.py --type rotation --resolution 1080p --backend taichi --hw gpu
  
  # Quick preview at 720p
  python generate_video.py --type rotation --frames 30 --resolution 720p --backend scipy
        """
    )
    
    parser.add_argument('--type', choices=['rotation', 'orbit', 'zoom'], 
                        default='rotation',
                        help='Type of animation (default: rotation)')
    parser.add_argument('--frames', type=int, default=60,
                        help='Number of frames to generate (default: 60)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for output video (default: 30)')
    parser.add_argument('--resolution', default='1080p',
                        choices=['480p', '720p', '1080p', '1440p', '4k'],
                        help='Output resolution (default: 1080p)')
    parser.add_argument('--backend', default='taichi',
                        choices=['scipy', 'numba', 'taichi', 'jax'],
                        help='Computational backend (default: taichi)')
    parser.add_argument('--hw', default='gpu',
                        choices=['cpu', 'gpu', 'cuda', 'vulkan'],
                        help='Hardware for taichi backend (default: gpu)')
    parser.add_argument('--output', type=str,
                        help='Output video filename (default: auto-generated)')
    parser.add_argument('--keep-frames', action='store_true',
                        help='Keep individual frame images after creating video')
    
    args = parser.parse_args()
    
    # Resolution mapping
    resolutions = {
        '480p': (854, 480),
        '720p': (1280, 720),
        '1080p': (1920, 1080),
        '1440p': (2560, 1440),
        '4k': (3840, 2160),
    }
    resolution = resolutions[args.resolution]
    
    # Output paths
    frames_dir = Path(f'frames_{args.type}_{args.resolution}')
    if args.output:
        output_video = Path(args.output)
    else:
        output_video = Path(f'blackhole_{args.type}_{args.resolution}_{args.fps}fps.mp4')
    
    # Generate frames
    print("=" * 70)
    print("BLACK HOLE VIDEO GENERATOR")
    print("=" * 70)
    print()
    
    if args.type == 'rotation':
        generate_rotation_frames(frames_dir, args.frames, resolution, 
                                 args.backend, args.hw, args.fps)
    elif args.type == 'orbit':
        generate_orbit_frames(frames_dir, args.frames, resolution,
                              args.backend, args.hw, args.fps)
    elif args.type == 'zoom':
        generate_zoom_frames(frames_dir, args.frames, resolution,
                             args.backend, args.hw, args.fps)
    
    # Create video
    success = create_video_from_frames(frames_dir, output_video, args.fps)
    
    # Cleanup frames
    if success and not args.keep-frames:
        import shutil
        print(f"\nCleaning up frames in {frames_dir}...")
        shutil.rmtree(frames_dir)
        print("✅ Cleanup complete")
    
    print()
    print("=" * 70)
    if success:
        print(f"✅ SUCCESS! Video saved: {output_video}")
    else:
        print(f"⚠️  Frames saved in: {frames_dir}")
        print("   Use ffmpeg manually to create video")
    print("=" * 70)


if __name__ == '__main__':
    main()
