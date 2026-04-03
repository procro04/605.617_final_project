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

Dependencies: numpy, matplotlib (both common; pip install matplotlib if needed)
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ── PGM reader ───────────────────────────────────────────────────────────────

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


# ── LIDAR data reader ────────────────────────────────────────────────────────

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


# ── Plotting ─────────────────────────────────────────────────────────────────

def make_figure(lidar_path: str, pgm_path: str):
    robot_x, robot_y, hit_x, hit_y = load_lidar(lidar_path)
    pgm = load_pgm(pgm_path)

    # Compute world-coordinate extents from the LIDAR data so the two panels
    # share the same spatial frame.
    all_x = np.concatenate([robot_x, hit_x])
    all_y = np.concatenate([robot_y, hit_y])
    margin = 2.0
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin

    # Build a 3-color colormap for the PGM: black=occupied, white=free, gray=unknown
    # PGM values: 0 → occupied, 128 → unknown, 255 → free
    grid_cmap = ListedColormap(["black", "gray", "white"])
    # Bin the PGM into 3 levels
    pgm_binned = np.zeros_like(pgm, dtype=np.uint8)
    pgm_binned[pgm == 0]   = 0   # occupied
    pgm_binned[pgm == 128] = 1   # unknown
    pgm_binned[pgm == 255] = 2   # free

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Occupancy Map Verification", fontsize=14, fontweight="bold")

    # ── Panel 1: Raw LIDAR data ──
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

    # ── Panel 2: PGM occupancy grid ──
    ax = axes[1]
    ax.set_title("Occupancy Grid (PGM)")
    ax.imshow(pgm_binned, cmap=grid_cmap, origin="upper",
              extent=[x_min, x_max, y_min, y_max])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")

    # ── Panel 3: Composite overlay ──
    ax = axes[2]
    ax.set_title("Overlay (Hits on Grid)")
    ax.imshow(pgm_binned, cmap=grid_cmap, origin="upper",
              extent=[x_min, x_max, y_min, y_max], alpha=0.6)
    ax.scatter(hit_x, hit_y, s=0.3, c="red", alpha=0.5)
    ax.plot(robot_x, robot_y, "-o", color="cyan", markersize=3,
            linewidth=1, alpha=0.8)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")

    plt.tight_layout()
    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Overlay LIDAR data on occupancy grid PGM for verification.")
    parser.add_argument("lidar", help="Path to LIDAR scan text file")
    parser.add_argument("pgm",   help="Path to occupancy grid .pgm file")
    parser.add_argument("--save", metavar="FILE",
                        help="Save figure to file instead of displaying")
    args = parser.parse_args()

    fig = make_figure(args.lidar, args.pgm)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
