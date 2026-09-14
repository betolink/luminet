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
    fps: int = 30,
    speed: float = 1.0,
    color_scheme: str = 'flux',
    bg_color: str = 'white',
    radial_res: int = 400,
    angular_res: int = 200
):
    """Generate frames showing black hole rotation (varying inclination).
    
    Args:
        output_dir: Directory to save frames
        n_frames: Number of frames to generate
        resolution: (width, height) in pixels
        backend: Computational backend ('scipy', 'taichi', etc.)
        hw: Hardware for taichi ('cpu' or 'gpu')
        fps: Target frames per second
        speed: Animation speed multiplier (higher = faster rotation)
        color_scheme: Color scheme ('flux', 'viridis', 'plasma', etc.)
        bg_color: Background color ('white' or 'black')
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup backend
    if backend == 'taichi':
        backend_obj = get_backend('taichi', arch=hw)
        bhmath._backend = backend_obj
    else:
        bhmath.set_backend(backend)
    
    # Inclination sweep: 0° to 90° (edge-on to face-on)
    # Speed multiplier affects the angle range covered
    incl_min = 0.1  # Almost face-on (rad)
    incl_max = np.pi / 2 - 0.1  # Almost edge-on (rad)
    
    # Adjust range based on speed (higher speed = larger angle change)
    angle_range = (incl_max - incl_min) * speed
    if angle_range > (incl_max - incl_min):
        # Loop through multiple rotations
        inclinations = np.linspace(incl_min, incl_min + angle_range, n_frames) % (incl_max - incl_min) + incl_min
    else:
        # Single sweep (partial or full)
        incl_end = min(incl_min + angle_range, incl_max)
        inclinations = np.linspace(incl_min, incl_end, n_frames)
    
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
            radial_resolution=radial_res,
            angular_resolution=angular_res
        )
        
        # Plot black hole with color scheme - let bh.plot create its own polar axis
        if color_scheme == 'flux':
            ax = bh.plot()  # Default flux coloring
        else:
            # Use custom colormap
            ax = bh.plot(cmap=color_scheme)
        
        # Get figure and set background colors
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        fig.set_dpi(dpi)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Add title with current angle (text color contrasts with background)
        title_color = 'black' if bg_color == 'white' else 'white'
        ax.set_title(f'Black Hole Inclination: {np.degrees(incl):.1f}°', 
                     fontsize=16, color=title_color)
        
        # Clean layout
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Save frame
        frame_path = output_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=dpi, facecolor=bg_color, edgecolor='none')
        plt.close(fig)
    
    print(f"\n✅ Generated {n_frames} frames in {output_dir}")
    return output_dir


def generate_orbit_frames(
    output_dir: Path,
    n_frames: int = 120,
    resolution: tuple = (1920, 1080),
    backend: str = 'taichi',
    hw: str = 'gpu',
    fps: int = 30,
    speed: float = 1.0,
    color_scheme: str = 'flux',
    bg_color: str = 'white',
    radial_res: int = 400,
    angular_res: int = 200
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
        speed: Animation speed multiplier
        color_scheme: Color scheme for black hole
        bg_color: Background color
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup backend
    if backend == 'taichi':
        backend_obj = get_backend('taichi', arch=hw)
        bhmath._backend = backend_obj
    else:
        bhmath.set_backend(backend)
    
    # Orbital motion: smooth sinusoidal variation
    # Speed affects how many orbits are completed
    base_incl = 1.2  # Base viewing angle (rad)
    incl_variation = 0.3  # Variation amplitude
    phases = np.linspace(0, 2 * np.pi * speed, n_frames)
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
            radial_resolution=radial_res,
            angular_resolution=angular_res
        )
        
        # Plot with color scheme - let bh.plot create its own polar axis
        if color_scheme == 'flux':
            ax = bh.plot()
        else:
            ax = bh.plot(cmap=color_scheme)
        
        # Get figure and set background colors
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        fig.set_dpi(dpi)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Add orbit progress indicator
        progress = (i / n_frames) * 360 * speed
        title_color = 'black' if bg_color == 'white' else 'white'
        ax.set_title(f'Orbital Position: {progress % 360:.0f}°', 
                     fontsize=16, color=title_color)
        
        ax.set_aspect('equal')
        ax.axis('off')
        # Clean layout - remove padding for proper scaling
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        frame_path = output_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=dpi, facecolor=bg_color, edgecolor='none')
        plt.close(fig)
    
    print(f"\n✅ Generated {n_frames} frames in {output_dir}")
    return output_dir


def generate_zoom_frames(
    output_dir: Path,
    n_frames: int = 60,
    resolution: tuple = (1920, 1080),
    backend: str = 'taichi',
    hw: str = 'gpu',
    fps: int = 30,
    color_scheme: str = 'flux',
    bg_color: str = 'white',
    radial_res: int = 400,
    angular_res: int = 200
):
    """Generate frames showing zoom in/out on black hole.
    
    Args:
        output_dir: Directory to save frames
        n_frames: Number of frames to generate
        resolution: (width, height) in pixels
        backend: Computational backend
        hw: Hardware for taichi
        fps: Target frames per second
        color_scheme: Color scheme for black hole
        bg_color: Background color
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
            outer_edge=20.0,
            radial_resolution=radial_res,
            angular_resolution=angular_res
        )
        
        # Plot with color scheme - let bh.plot create its own polar axis
        if color_scheme == 'flux':
            ax = bh.plot()
        else:
            ax = bh.plot(cmap=color_scheme)
        
        # Get figure and set background colors
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        fig.set_dpi(dpi)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Maintain consistent axis limits for smooth zoom (polar: radius only)
        ax.set_ylim((0, radius_max * 1.2))
        
        title_color = 'black' if bg_color == 'white' else 'white'
        ax.set_title(f'Disk Radius: {radius:.1f}M', 
                     fontsize=16, color=title_color)
        
        ax.set_aspect('equal')
        ax.axis('off')
        # Clean layout - remove padding for proper scaling
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        frame_path = output_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=dpi, facecolor=bg_color, edgecolor='none')
        plt.close(fig)
    
    print(f"\n✅ Generated {n_frames} frames in {output_dir}")
    return output_dir


def generate_azimuth_frames(
    output_dir: Path,
    n_frames: int = 60,
    resolution: tuple = (1920, 1080),
    backend: str = 'taichi',
    hw: str = 'gpu',
    fps: int = 30,
    speed: float = 1.0,  # Number of full 360° rotations
    color_scheme: str = 'flux',
    bg_color: str = 'white',
    radial_res: int = 400,
    angular_res: int = 200,
    inclination: float = np.radians(20)  # Fixed viewing angle from face-on (0°)
):
    """Generate frames showing camera rotation around black hole (azimuth animation).
    
    Unlike rotation (which changes viewing angle), azimuth animation keeps
    viewing angle fixed while rotating around the black hole.
    
    Args:
        output_dir: Directory to save frames
        n_frames: Number of frames to generate
        resolution: (width, height) in pixels
        backend: Computational backend
        hw: Hardware for taichi backend
        fps: Target frames per second
        speed: Number of complete 360° rotations to animate
        color_scheme: Color scheme for black hole
        bg_color: Background color
        radial_res: Radial resolution for smoothness
        angular_res: Angular resolution
        inclination: Fixed viewing angle in radians (default: 20° from disk plane)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup backend
    if backend == 'taichi':
        backend_obj = get_backend('taichi', arch=hw)
        bhmath._backend = backend_obj
    else:
        bhmath.set_backend(backend)
    
    # Figure setup
    dpi = 100
    figsize = (resolution[0] / dpi, resolution[1] / dpi)
    
    print(f"Generating {n_frames} azimuth frames at {resolution[0]}×{resolution[1]}...")
    print(f"Backend: {backend} ({hw if backend == 'taichi' else 'N/A'})")
    print(f"Fixed inclination: {np.degrees(inclination):.1f}°")
    print(f"Azimuth range: 0° → {360 * speed}°")
    print()
    
    # Azimuth varies from 0 to 2π * speed
    azimuth_angles = np.linspace(0, 2 * np.pi * speed, n_frames)
    
    for i, azimuth in enumerate(tqdm(azimuth_angles, desc="Rendering frames")):
        # Create black hole with FIXED inclination
        bh = BlackHole(
            mass=1.0,
            incl=inclination,  # Fixed viewing angle
            acc=1.0,
            outer_edge=20.0,
            radial_resolution=radial_res,
            angular_resolution=angular_res
        )
        
        # Plot with color scheme - let bh.plot create its own polar axis
        if color_scheme == 'flux':
            ax = bh.plot()
        else:
            ax = bh.plot(cmap=color_scheme)
        
        # Get figure and set background colors
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        fig.set_dpi(dpi)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Rotate entire plot to create azimuth animation
        # Set theta offset to rotate around black hole
        ax.set_theta_offset(azimuth)
        
        # Add title with current azimuth angle
        title_color = 'black' if bg_color == 'white' else 'white'
        azimuth_degrees = np.degrees(azimuth) % 360
        ax.set_title(f'Azimuth: {azimuth_degrees:.0f}°', 
                     fontsize=16, color=title_color)
        
        # Clean layout
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Save frame
        frame_path = output_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=dpi, facecolor=bg_color, edgecolor='none')
        plt.close(fig)
    
    print(f"\n✅ Generated {n_frames} frames in {output_dir}")
    return output_dir


def generate_combined_frames(
    output_dir: Path,
    n_frames: int = 120,
    resolution: tuple = (1920, 1080),
    backend: str = 'taichi',
    hw: str = 'gpu',
    fps: int = 30,
    speed: float = 1.0,
    azimuth_speed: float = 1.0,
    color_scheme: str = 'flux',
    bg_color: str = 'white',
    radial_res: int = 400,
    angular_res: int = 200
):
    """Generate frames combining orbital motion with azimuth rotation.
    
    Camera orbits around black hole (azimuth rotation) while viewing angle
    oscillates (inclination variation), creating complex orbital paths.
    
    Args:
        output_dir: Directory to save frames
        n_frames: Number of frames to generate
        resolution: (width, height) in pixels
        backend: Computational backend
        hw: Hardware for taichi
        fps: Target frames per second
        speed: Animation speed for inclination oscillation
        azimuth_speed: Speed of azimuth rotation (1.0 = 1 full circle)
        color_scheme: Color scheme for black hole
        bg_color: Background color
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup backend
    if backend == 'taichi':
        backend_obj = get_backend('taichi', arch=hw)
        bhmath._backend = backend_obj
    else:
        bhmath.set_backend(backend)
    
    # Combined motion: sinusoidal inclination + azimuth rotation
    base_incl = 1.2
    incl_variation = 0.3
    phases = np.linspace(0, 2 * np.pi * speed, n_frames)
    inclinations = base_incl + incl_variation * np.sin(phases)
    
    # Azimuth varies from 0 to 2π * azimuth_speed
    azimuth_angles = np.linspace(0, 2 * np.pi * azimuth_speed, n_frames)
    
    # Figure setup
    dpi = 100
    figsize = (resolution[0] / dpi, resolution[1] / dpi)
    
    print(f"Generating {n_frames} combined frames at {resolution[0]}×{resolution[1]}...")
    print(f"Backend: {backend} ({hw if backend == 'taichi' else 'N/A'})")
    print(f"Inclination: {np.degrees(base_incl):.1f}° ± {np.degrees(incl_variation):.1f}°")
    print(f"Azimuth: {360 * azimuth_speed:.0f}° rotation")
    print()
    
    for i, (incl, azimuth) in enumerate(tqdm(zip(inclinations, azimuth_angles), desc="Rendering frames", total=n_frames)):
        bh = BlackHole(
            mass=1.0,
            incl=incl,
            acc=1.0,
            outer_edge=20.0,
            radial_resolution=radial_res,
            angular_resolution=angular_res
        )
        
        if color_scheme == 'flux':
            ax = bh.plot()
        else:
            ax = bh.plot(cmap=color_scheme)
        
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        fig.set_dpi(dpi)
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Rotate entire plot for azimuth animation
        ax.set_theta_offset(azimuth)
        
        title_color = 'black' if bg_color == 'white' else 'white'
        azimuth_degrees = np.degrees(azimuth) % 360
        incl_degrees = np.degrees(incl)
        ax.set_title(f'Azimuth: {azimuth_degrees:.0f}° | Inclination: {incl_degrees:.1f}°', 
                     fontsize=16, color=title_color)
        
        ax.set_aspect('equal')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        frame_path = output_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=dpi, facecolor=bg_color, edgecolor='none')
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
  
  # Faster rotation (2x speed)
  python generate_video.py --type rotation --speed 2.0
  
  # Different color schemes
  python generate_video.py --type rotation --color-scheme viridis
  python generate_video.py --type rotation --color-scheme plasma
  python generate_video.py --type rotation --color-scheme hot
  
  # Black background
  python generate_video.py --type rotation --bg-color black
  
  # Orbital animation (smooth orbit)
  python generate_video.py --type orbit --frames 120 --fps 30
  
   # Zoom animation
   python generate_video.py --type zoom --frames 60
   
    # Azimuth animation (camera rotation around black hole)
    python generate_video.py --type azimuth --frames 300 --inclination 20
    
    # Combined orbit + azimuth (complex orbital path)
    python generate_video.py --type combined --frames 180 --azimuth-speed 2.0
    
    # High quality 1080p with GPU and custom colors
   python generate_video.py --type rotation --resolution 1080p --backend taichi --hw gpu --color-scheme inferno
  
  # Quick preview at 720p
  python generate_video.py --type rotation --frames 30 --resolution 720p --backend scipy
        """
     )
    
    parser.add_argument('--type', choices=['rotation', 'orbit', 'zoom', 'azimuth', 'combined'], 
                        default='rotation',
                        help='Type of animation (default: rotation)')
    parser.add_argument('--frames', type=int, default=60,
                        help='Number of frames to generate (default: 60)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for output video (default: 30)')
    parser.add_argument('--resolution', default='1080p',
                        choices=['480p', '720p', '1080p', '1440p', '4k'],
                        help='Output resolution (default: 1080p)')
    parser.add_argument('--radial-res', type=int, default=400,
                        help='Radial resolution for smoothness (default: 400)')
    parser.add_argument('--angular-res', type=int, default=200,
                        help='Angular resolution (default: 200)')
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
    
    # Animation control
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Animation speed multiplier for inclination (default: 1.0, higher=faster)')
    parser.add_argument('--azimuth-speed', type=float, default=1.0,
                        help='Azimuth rotation speed (default: 1.0, number of full circles)')
    parser.add_argument('--color-scheme', '--cmap', default='flux',
                        choices=['flux', 'viridis', 'plasma', 'inferno', 'hot', 'cool', 'rainbow', 'jet', 'Greys_r'],
                        help='Color scheme for black hole (default: flux)')
    parser.add_argument('--bg-color', default='white',
                        choices=['white', 'black'],
                        help='Background color (default: white)')
    parser.add_argument('--start-angle', type=float, default=None,
                        help='Start angle for rotation/azimuth (degrees)')
    parser.add_argument('--end-angle', type=float, default=None,
                        help='End angle for rotation/azimuth (degrees)')
    parser.add_argument('--inclination', type=float, default=20.0,
                        help='Fixed viewing angle for azimuth animation (degrees, default: 20)')
    
    args = parser.parse_args()
    
    # Resolution mapping (all dimensions must be even for h264)
    resolutions = {
        '480p': (856, 480),   # Standard 480p (16:9, width adjusted for even number and DPI rounding)
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
                                 args.backend, args.hw, args.fps,
                                 args.speed, args.color_scheme, args.bg_color,
                                 args.radial_res, args.angular_res)
    elif args.type == 'orbit':
        generate_orbit_frames(frames_dir, args.frames, resolution,
                               args.backend, args.hw, args.fps,
                               args.speed, args.color_scheme, args.bg_color,
                               args.radial_res, args.angular_res)
    elif args.type == 'zoom':
        generate_zoom_frames(frames_dir, args.frames, resolution,
                              args.backend, args.hw, args.fps,
                              args.color_scheme, args.bg_color,
                              args.radial_res, args.angular_res)
    elif args.type == 'azimuth':
        generate_azimuth_frames(frames_dir, args.frames, resolution,
                                 args.backend, args.hw, args.fps,
                                 args.speed, args.color_scheme, args.bg_color,
                                 args.radial_res, args.angular_res,
                                 np.radians(args.inclination))
    elif args.type == 'combined':
        generate_combined_frames(frames_dir, args.frames, resolution,
                                  args.backend, args.hw, args.fps,
                                  args.speed, args.azimuth_speed, args.color_scheme, args.bg_color,
                                  args.radial_res, args.angular_res)
    
    # Create video
    success = create_video_from_frames(frames_dir, output_video, args.fps)
    
    # Cleanup frames
    if success and not args.keep_frames:
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
