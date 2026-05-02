#!/usr/bin/env python3
"""
visualize_map.py

Verification tool: overlays raw LIDAR scan data on top of the occupancy grid
PGM output so you can visually confirm the map matches the input.

Produces a 3-panel figure:
  1. Raw LIDAR hits (scatter) + robot path
  2. Occupancy grid PGM image
  3. Composite overlay — LIDAR hits on top of the grid

Usage:
    python3 visualize_map.py <lidar_data.txt> <occupancy.pgm> [--save output.png]
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# PGM reader

def load_pgm(filepath: str) -> np.ndarray:
    """Load a P5 (binary) PGM file and return it as a 2D uint8 array."""
    with open(filepath, "rb") as f:
        # Magic number
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Not a P5 PGM file (got {magic!r})")

        # Skip comments
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()

        width, height = map(int, line.split())
        max_val = int(f.readline().strip())
        assert max_val == 255, f"Expected maxval 255, got {max_val}"

        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape((height, width))


# LIDAR data reader

def load_lidar(filepath: str):
    """
    Returns (robot_xs, robot_ys, hit_xs, hit_ys) as flat numpy arrays.
    Skips comment lines starting with '#'.
    """
    robot_x, robot_y = [], []
    hit_x, hit_y = [], []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = list(map(float, line.split()))
            if len(vals) < 5:
                continue  # need at least pose + one hit

            rx, ry, _theta = vals[0], vals[1], vals[2]
            robot_x.append(rx)
            robot_y.append(ry)

            coords = vals[3:]
            for i in range(0, len(coords) - 1, 2):
                hit_x.append(coords[i])
                hit_y.append(coords[i + 1])

    return (np.array(robot_x), np.array(robot_y),
            np.array(hit_x),   np.array(hit_y))


# Plotting

def bin_pgm(pgm: np.ndarray) -> np.ndarray:
    """Map PGM pixel values (0=occupied, 128=unknown, 255=free) to 3-level array."""
    binned = np.zeros_like(pgm, dtype=np.uint8)
    binned[pgm == 128] = 1
    binned[pgm == 255] = 2
    return binned


def pgm_extent_from_lidar(robot_x, robot_y, hit_x, hit_y, margin=1.0):
    """Estimate world-coordinate extent of the PGM from LIDAR data + margin."""
    all_x = np.concatenate([robot_x, hit_x])
    all_y = np.concatenate([robot_y, hit_y])
    return (all_x.min() - margin, all_x.max() + margin,
            all_y.min() - margin, all_y.max() + margin)


def pgm_extent_from_grid(pgm: np.ndarray, grid_origin_x: float,
                           grid_origin_y: float, resolution: float):
    """
    Compute exact world-coordinate extent of a PGM given the grid's center
    origin and cell resolution (metres/cell).

    The OccupancyGrid is built so that (grid_origin_x, grid_origin_y) is the
    centre of the grid.  The PGM is written row-major with row 0 at the top,
    meaning row 0 corresponds to the *maximum* world-y edge of the grid.
    """
    height, width = pgm.shape
    half_w = width  * resolution / 2.0
    half_h = height * resolution / 2.0
    x_min = grid_origin_x - half_w
    x_max = grid_origin_x + half_w
    # imshow origin="upper": row 0 → top of image → y_max in world coords
    y_min = grid_origin_y - half_h
    y_max = grid_origin_y + half_h
    return x_min, x_max, y_min, y_max


def draw_panels(fig, axes, robot_x, robot_y, hit_x, hit_y,
                 pgm_binned, x_min, x_max, y_min, y_max, title="Occupancy Map Verification"):
    grid_cmap = ListedColormap(["black", "gray", "white"])
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.set_title("Raw LIDAR Input")
    ax.scatter(hit_x, hit_y, s=0.3, c="red", alpha=0.4, label="Hits")
    ax.plot(robot_x, robot_y, "-o", color="blue", markersize=3,
            linewidth=1, alpha=0.7, label="Robot path")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax = axes[1]
    ax.set_title("Occupancy Grid (PGM)")
    # imshow with origin="upper" places row 0 at the top (y_max in world coords).
    ax.imshow(pgm_binned, cmap=grid_cmap, origin="upper",
              extent=[x_min, x_max, y_min, y_max])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")


def make_figure(lidar_path: str, pgm_path: str, resolution: float = 0.05):
    """Figure for test datasets: derives grid origin from LIDAR extents (mirrors CUDA logic)."""
    robot_x, robot_y, hit_x, hit_y = load_lidar(lidar_path)
    pgm = load_pgm(pgm_path)
    pgm_binned = bin_pgm(pgm)

    # Replicate the grid-center computation from occupancy_map_cuda.cu main()
    all_x = np.concatenate([robot_x, hit_x])
    all_y = np.concatenate([robot_y, hit_y])
    cx = (all_x.min() + all_x.max()) / 2.0
    cy = (all_y.min() + all_y.max()) / 2.0
    x_min, x_max, y_min, y_max = pgm_extent_from_grid(pgm, cx, cy, resolution)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    draw_panels(fig, axes, robot_x, robot_y, hit_x, hit_y,
                 pgm_binned, x_min, x_max, y_min, y_max)
    plt.tight_layout()
    return fig


def make_figure_intel(lidar_path: str, pgm_path: str,
                      grid_origin_x: float, grid_origin_y: float,
                      resolution: float):
    """
    Figure for the Intel Research Lab dataset.

    Uses exact grid origin + resolution to align the PGM with world coordinates
    instead of approximating from LIDAR extents.  Pass the same origin and
    resolution that were given to the occupancy_map_cuda binary.
    """
    robot_x, robot_y, hit_x, hit_y = load_lidar(lidar_path)
    pgm = load_pgm(pgm_path)
    pgm_binned = bin_pgm(pgm)
    x_min, x_max, y_min, y_max = pgm_extent_from_grid(
        pgm, grid_origin_x, grid_origin_y, resolution)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    draw_panels(fig, axes, robot_x, robot_y, hit_x, hit_y,
                 pgm_binned, x_min, x_max, y_min, y_max,
                 title="Occupancy Map Verification (Intel Dataset)")
    plt.tight_layout()
    return fig

def make_figure_lidar_only(lidar_path: str):
    """Figure for just LIDAR scan data — no PGM required."""
    robot_x, robot_y, hit_x, hit_y = load_lidar(lidar_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Raw LIDAR Input", fontsize=14, fontweight="bold")

    ax.scatter(hit_x, hit_y, s=0.3, c="red", alpha=0.4, label="Hits")
    ax.plot(robot_x, robot_y, "-o", color="blue", markersize=3,
            linewidth=1, alpha=0.7, label="Robot path")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LIDAR scan data, optionally overlaid on an occupancy grid PGM.")
    parser.add_argument("lidar", help="Path to LIDAR scan text file")
    parser.add_argument("pgm", nargs="?", default=None,
                        help="Path to occupancy grid .pgm file (optional)")
    parser.add_argument("--save", metavar="FILE",
                        help="Save figure to file instead of displaying")
    parser.add_argument("--grid-origin", nargs=2, type=float, metavar=("CX", "CY"),
                        help="World-frame centre of the occupancy grid (metres). "
                             "Enables exact PGM alignment (use for Intel dataset).")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="Grid cell size in metres/cell (default 0.05, required with --grid-origin).")
    args = parser.parse_args()

    if args.pgm is None:
        fig = make_figure_lidar_only(args.lidar)
    elif args.grid_origin is not None:
        cx, cy = args.grid_origin
        fig = make_figure_intel(args.lidar, args.pgm, cx, cy, args.resolution)
    else:
        fig = make_figure(args.lidar, args.pgm, resolution=args.resolution)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
