// occupancy_map_cpu.cpp
// CPU-only occupancy grid mapping implementation.
// Ported from Python QuadMap (mapping.py) to a flat 2D grid suitable for
// future CUDA acceleration.
// Usage:
//   ./occupancy_map_cpu <lidar_data.txt> [grid_size] [resolution]
//   lidar_data.txt  — Intel Research Lab format: each line is
//                    "x_robot y_robot theta  x1 y1  x2 y2 ... xN yN"
//                    (the first 3 values are the robot pose, followed by
//                    pairs of LIDAR hit coordinates in world frame)
//   grid_size       — physical side length in meters (default: 40.0)
//   resolution      — meters per cell (default: 0.05)
// Outputs:
//   occupancy_####.pgm  — grayscale image per scan (or final only, configurable)
//   timing_report.txt   — per-stage benchmark results*

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "CodeTimer.h"

// ---------------------------------------------------------------------------
// Configuration & Constants
// ---------------------------------------------------------------------------

// Cell states — matches Python MapType. Using int8_t so the grid is compact.
// These values also double as PGM pixel intensities after remapping.
enum CellState : int8_t
{
    UNKNOWN = -1,
    UNOCCUPIED = 0,
    OCCUPIED = 1
};

// ---------------------------------------------------------------------------
// LIDAR Scan Data
// ---------------------------------------------------------------------------
struct Pose2D
{
    float x;
    float y;
    float theta;
};

struct LidarScan
{
    Pose2D pose;
    std::vector<std::pair<float, float>> hits; // (x, y) in world frame
};

// Parse the Intel Research Lab LIDAR dataset.
// Expected line format:  x_robot  y_robot  theta  x1 y1  x2 y2  ...
// Lines starting with '#' are comments.
std::vector<LidarScan> load_lidar_dataset(const std::string &filepath)
{
    std::vector<LidarScan> scans;
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open " << filepath << "\n";
        return scans;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        LidarScan scan;

        // First 3 values: robot pose
        if (!(iss >> scan.pose.x >> scan.pose.y >> scan.pose.theta))
            continue;

        // Remaining pairs: LIDAR hit points (x, y) in world frame
        float hx, hy;
        while (iss >> hx >> hy)
        {
            scan.hits.emplace_back(hx, hy);
        }

        if (!scan.hits.empty())
        {
            scans.push_back(std::move(scan));
        }
    }

    std::cout << "Loaded " << scans.size() << " scans from " << filepath << "\n";
    return scans;
}

// ---------------------------------------------------------------------------
// Occupancy Grid  (flat, CUDA-friendly)
// ---------------------------------------------------------------------------

class OccupancyGrid
{
public:
    // grid_size_meters  Physical side length of the square grid.
    // resolution        Meters per grid cell.
    // origin_x/y        World coordinate of the grid center.
    OccupancyGrid(float grid_size_meters, float resolution,
                  float origin_x = 0.0f, float origin_y = 0.0f)
        : grid_size_(grid_size_meters),
          resolution_(resolution),
          origin_x_(origin_x),
          origin_y_(origin_y)
    {
        num_cells_ = static_cast<int>(std::ceil(grid_size_ / resolution_));
        cells_.resize(static_cast<size_t>(num_cells_) * num_cells_, UNKNOWN);
        std::cout << "Grid: " << num_cells_ << " x " << num_cells_
                  << " cells  (" << cells_.size() * sizeof(int8_t) << " bytes)\n";
    }

    int num_cells() const { return num_cells_; }
    float resolution() const { return resolution_; }
    float grid_size() const { return grid_size_; }
    int8_t *data() { return cells_.data(); }
    const int8_t *data() const { return cells_.data(); }

    // Convert world (x, y) -> grid (col, row). Returns false if out of bounds.
    bool world_to_grid(float wx, float wy, int &col, int &row) const
    {
        float local_x = wx - (origin_x_ - grid_size_ / 2.0f);
        float local_y = wy - (origin_y_ - grid_size_ / 2.0f);
        col = static_cast<int>(std::floor(local_x / resolution_));
        row = static_cast<int>(std::floor(local_y / resolution_));
        return (col >= 0 && col < num_cells_ && row >= 0 && row < num_cells_);
    }

    // Flat index from (col, row).
    inline int idx(int col, int row) const { return row * num_cells_ + col; }

    // Set a cell's state (bounds-checked).
    void set_cell(int col, int row, CellState state)
    {
        if (col >= 0 && col < num_cells_ && row >= 0 && row < num_cells_)
        {
            cells_[idx(col, row)] = state;
        }
    }

    CellState get_cell(int col, int row) const
    {
        if (col >= 0 && col < num_cells_ && row >= 0 && row < num_cells_)
        {
            return static_cast<CellState>(cells_[idx(col, row)]);
        }
        return UNKNOWN;
    }

    // Core Update Functions
    // These are written as free-standing-style logic on raw pointers so
    // they translate directly to CUDA kernels later.

    // Mark a single cell as OCCUPIED (the endpoint of a ray).
    void point_update(float wx, float wy, CellState state)
    {
        int col, row;
        if (world_to_grid(wx, wy, col, row))
        {
            cells_[idx(col, row)] = state;
        }
    }

    // Bresenham ray trace from robot position to hit point.
    // Marks all cells along the ray as UNOCCUPIED, then marks the endpoint
    // as OCCUPIED.
    //
    // This is the function you'll later split across CPU (ray trace) and
    // GPU (endpoint marking) in your first CUDA pass, then fully port to
    // GPU with atomics in a later pass.
    void ray_update(float ox, float oy, float hx, float hy)
    {
        int c0, r0, c1, r1;
        if (!world_to_grid(ox, oy, c0, r0))
            return;
        if (!world_to_grid(hx, hy, c1, r1))
            return;

        // Bresenham line from (c0,r0) to (c1,r1)
        bresenham_ray(cells_.data(), num_cells_, c0, r0, c1, r1);
    }

    // Process an entire LIDAR scan (all rays from the robot pose).
    void process_scan(const LidarScan &scan)
    {
        for (const auto &[hx, hy] : scan.hits)
        {
            ray_update(scan.pose.x, scan.pose.y, hx, hy);
        }
    }

    // Write the grid to a PGM (Portable Gray Map) image file.
    // UNKNOWN -> gray (128), UNOCCUPIED -> white (255), OCCUPIED -> black (0).
    bool write_pgm(const std::string &filepath) const
    {
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs)
            return false;

        ofs << "P5\n"
            << num_cells_ << " " << num_cells_ << "\n255\n";
        for (int r = num_cells_ - 1; r >= 0; --r)
        { // flip Y so up=north
            for (int c = 0; c < num_cells_; ++c)
            {
                int8_t v = cells_[idx(c, r)];
                uint8_t pixel;
                switch (v)
                {
                case OCCUPIED:
                    pixel = 0;
                    break;
                case UNOCCUPIED:
                    pixel = 255;
                    break;
                default:
                    pixel = 128;
                    break; // UNKNOWN
                }
                ofs.put(static_cast<char>(pixel));
            }
        }
        return true;
    }

private:
    float grid_size_;
    float resolution_;
    float origin_x_;
    float origin_y_;
    int num_cells_;

    // Flat, contiguous cell storage — row-major.  This is what gets
    // cudaMemcpy'd to the device in the CUDA version.
    std::vector<int8_t> cells_;

    // Bresenham (static so it can become a __device__ function)

    // Bresenham line rasterization.  Marks every cell along the ray as
    // UNOCCUPIED, then marks the final cell (c1, r1) as OCCUPIED.
    //
    // grid       Pointer to the flat grid array.
    // width      Number of cells per row (grid is width x width).
    // c0,r0      Start cell (robot position).
    // c1,r1      End cell (LIDAR hit).
    static void bresenham_ray(int8_t *grid, int width,
                              int c0, int r0, int c1, int r1)
    {
        // delta column, delta row
        int dc = std::abs(c1 - c0);
        int dr = std::abs(r1 - r0);
        // step column, step row
        // Figure out which way we need to step based on our start and end
        // coordinates
        int sc = (c0 < c1) ? 1 : -1;
        int sr = (r0 < r1) ? 1 : -1;
        int err = dc - dr;

        int c = c0, r = r0;

        while (true)
        {
            // If we've reached the endpoint, mark OCCUPIED and stop.
            if (c == c1 && r == r1)
            {
                grid[r * width + c] = OCCUPIED;
                break;
            }

            // All intermediate cells are free space.
            grid[r * width + c] = UNOCCUPIED;

            // Multiply error by 2 to keep easy integer math
            int e2 = 2 * err;
            // Adjust x and error
            if (e2 > -dr)
            {
                err -= dr;
                c += sc;
            }
            // Adjust y and error
            if (e2 < dc)
            {
                err += dc;
                r += sr;
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void print_usage(const char *prog)
{
    std::cout << "Usage: " << prog
              << " <lidar_data.txt> [grid_size_m] [resolution_m]\n"
              << "  grid_size_m   — physical side length (default 40.0)\n"
              << "  resolution_m  — meters per cell      (default 0.05)\n";
}

int main(int argc, char *argv[])
{
    // Parse arguments ----------------------------------------------------
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::string data_file = argv[1];
    float grid_size = (argc > 2) ? std::stof(argv[2]) : 40.0f;
    float resolution = (argc > 3) ? std::stof(argv[3]) : 0.05f;

    std::vector<TimingRecord> timing_log;

    // Load dataset -------------------------------------------------------
    std::vector<LidarScan> scans;
    {
        ScopedTimer t("Load dataset", timing_log);
        scans = load_lidar_dataset(data_file);
    }
    if (scans.empty())
    {
        std::cerr << "No scans loaded. Exiting.\n";
        return 1;
    }

    // Compute a reasonable grid origin from the data ---------------------
    float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;
    for (const auto &s : scans)
    {
        min_x = std::min(min_x, s.pose.x);
        max_x = std::max(max_x, s.pose.x);
        min_y = std::min(min_y, s.pose.y);
        max_y = std::max(max_y, s.pose.y);
        for (const auto &[hx, hy] : s.hits)
        {
            min_x = std::min(min_x, hx);
            max_x = std::max(max_x, hx);
            min_y = std::min(min_y, hy);
            max_y = std::max(max_y, hy);
        }
    }
    float cx = (min_x + max_x) / 2.0f;
    float cy = (min_y + max_y) / 2.0f;
    // Ensure the grid is large enough to cover all data, with some margin
    float data_span = std::max(max_x - min_x, max_y - min_y) + 4.0f;
    if (grid_size < data_span)
    {
        grid_size = data_span;
        std::cout << "Auto-expanded grid_size to " << grid_size << " m to fit data\n";
    }
    std::cout << "Data extents: x=[" << min_x << ", " << max_x
              << "]  y=[" << min_y << ", " << max_y << "]\n";
    std::cout << "Grid center: (" << cx << ", " << cy << ")\n";

    // Initialize grid ----------------------------------------------------
    OccupancyGrid grid(grid_size, resolution, cx, cy);

    // Process all scans --------------------------------------------------
    {
        ScopedTimer t("Process all scans (total)", timing_log);

        for (size_t i = 0; i < scans.size(); ++i)
        {
            ScopedTimer ts("Scan #" + std::to_string(i), timing_log);
            grid.process_scan(scans[i]);
        }
    }

    // Write final output -------------------------------------------------
    {
        ScopedTimer t("Write PGM output", timing_log);
        std::string outfile = "occupancy_final.pgm";
        if (grid.write_pgm(outfile))
        {
            std::cout << "Wrote " << outfile << "\n";
        }
        else
        {
            std::cerr << "Failed to write " << outfile << "\n";
        }
    }

    // Write timing report
    std::ofstream rpt("timing_report.txt");
    rpt << "=== Occupancy Map CPU Timing Report ===\n\n";
    rpt << "Grid: " << grid.num_cells() << " x " << grid.num_cells()
        << "  resolution=" << grid.resolution() << " m\n";
    rpt << "Scans processed: " << scans.size() << "\n\n";

    // Print summary totals first
    double total_scan_ms = 0.0;
    for (const auto &r : timing_log)
    {
        if (r.label.rfind("Scan #", 0) == 0)
            total_scan_ms += r.elapsed_ms;
    }

    // Print high-level timings first.
    for (const auto &r : timing_log)
    {
        // Only print aggregate records and non-per-scan, non-substep records.
        if (r.label.rfind("Scan #", 0) != 0 &&
            r.label.rfind("  ", 0) != 0)
        {
            print_timing_line(rpt, r.label, r.elapsed_ms);
        }
    }

    // Per-scan average.
    if (!scans.empty())
    {
        double avg = total_scan_ms / scans.size();
        std::cout << "\n";
        rpt << "\n";
        print_timing_line(rpt, "Average per scan", avg);
    }

    std::cout << "\nFull timing log: timing_report.txt\n";

    return 0;
}
