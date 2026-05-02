#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

#include <cuda_runtime.h>

#include "OccupanyGrid.h"
#include "CudaHelpers.h"
#include "occupancy_kernels.cuh"

// Parse the Intel Research Lab LIDAR dataset.
// Expected line format:  x_robot  y_robot  theta  x1 y1  x2 y2
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

// grid_size_meters  Physical side length of the square grid.
// resolution        Meters per grid cell.
// origin_x/y        World coordinate of the grid center.
OccupancyGrid::OccupancyGrid(float grid_size_meters, float resolution,
                             float origin_x, float origin_y)
    : grid_size_(grid_size_meters),
      resolution_(resolution),
      origin_x_(origin_x),
      origin_y_(origin_y),
      d_grid_(nullptr),
      d_hits_x_(nullptr),
      d_hits_y_(nullptr),
      hits_capacity_(0)
{
    num_cells_ = static_cast<int>(std::ceil(grid_size_ / resolution_));
    size_t grid_bytes = static_cast<size_t>(num_cells_) * num_cells_ * sizeof(int8_t);
    cells_.resize(static_cast<size_t>(num_cells_) * num_cells_, UNKNOWN);

    // Allocate device grid and initialize to UNKNOWN.
    CUDA_CHECK(cudaMalloc(&d_grid_, grid_bytes));
    CUDA_CHECK(cudaMemcpy(d_grid_, cells_.data(), grid_bytes,
                          cudaMemcpyHostToDevice));

    std::cout << "Grid: " << num_cells_ << " x " << num_cells_
              << " cells  (" << grid_bytes << " bytes)\n";
}

OccupancyGrid::~OccupancyGrid()
{
    if (d_grid_)
    {
        cudaFree(d_grid_);
        d_grid_ = nullptr;
    }
    if (d_hits_x_)
    {
        cudaFree(d_hits_x_);
        cudaFree(d_hits_y_);
        d_hits_x_ = d_hits_y_ = nullptr;
    }
}

// Convert world (x, y) -> grid (col, row). Returns false if out of bounds.
bool OccupancyGrid::world_to_grid(float wx, float wy, int &col, int &row) const
{
    float local_x = wx - (origin_x_ - grid_size_ / 2.0f);
    float local_y = wy - (origin_y_ - grid_size_ / 2.0f);
    col = static_cast<int>(std::floor(local_x / resolution_));
    row = static_cast<int>(std::floor(local_y / resolution_));
    return (col >= 0 && col < num_cells_ && row >= 0 && row < num_cells_);
}

// Set a cell's state (bounds-checked).
void OccupancyGrid::set_cell(int col, int row, CellState state)
{
    if (col >= 0 && col < num_cells_ && row >= 0 && row < num_cells_)
    {
        cells_[idx(col, row)] = state;
    }
}

CellState OccupancyGrid::get_cell(int col, int row) const
{
    if (col >= 0 && col < num_cells_ && row >= 0 && row < num_cells_)
    {
        return static_cast<CellState>(cells_[idx(col, row)]);
    }
    return UNKNOWN;
}

// Mark a single cell as OCCUPIED (the endpoint of a ray).
void OccupancyGrid::point_update(float wx, float wy, CellState state)
{
    int col, row;
    if (world_to_grid(wx, wy, col, row))
    {
        cells_[idx(col, row)] = state;
    }
}

// Process an entire LIDAR scan entirely on the GPU.
// 1) Copy hit coordinates to the device.
// 2) Kernel 1: mark all endpoints OCCUPIED (no atomics needed).
// 3) Kernel 2: Bresenham ray trace for free space using atomicCAS.
// The device grid is NOT synced back to the host between scans; the
// host copy is only updated when sync_from_device() is called.
void OccupancyGrid::process_scan_cuda(const LidarScan &scan,
                                      std::vector<TimingRecord> &timing_log)
{
    int num_hits = static_cast<int>(scan.hits.size());
    if (num_hits == 0)
        return;

    // Grow the persistent device hit buffers only when this scan is larger
    // than any previous one.  cudaMalloc/cudaFree is called at most once per
    // new high-water mark rather than every scan.
    if (num_hits > hits_capacity_)
    {
        if (d_hits_x_)
        {
            CUDA_CHECK(cudaFree(d_hits_x_));
            CUDA_CHECK(cudaFree(d_hits_y_));
        }
        CUDA_CHECK(cudaMalloc(&d_hits_x_, num_hits * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hits_y_, num_hits * sizeof(float)));
        hits_capacity_ = num_hits;
    }

    // Pack hit coordinates into flat arrays for the GPU.
    std::vector<float> hits_x(num_hits);
    std::vector<float> hits_y(num_hits);
    for (int i = 0; i < num_hits; ++i)
    {
        hits_x[i] = scan.hits[i].first;
        hits_y[i] = scan.hits[i].second;
    }

    size_t hits_bytes = num_hits * sizeof(float);

    // Copy hits to device.
    {
        ScopedTimer t("  H2D hit data", timing_log);
        CUDA_CHECK(cudaMemcpy(d_hits_x_, hits_x.data(), hits_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_hits_y_, hits_y.data(), hits_bytes,
                              cudaMemcpyHostToDevice));
    }

    int num_blocks = (num_hits + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch both kernels back-to-back on the default stream (stream 0).
    // The stream guarantees kernel 2 starts only after kernel 1 finishes,
    // so no intermediate sync is needed between them.  A single sync after
    // both kernels is sufficient for timing and error checking.
    {
        ScopedTimer t("  GPU kernels (mark+trace)", timing_log);

        // Kernel 1: mark endpoints OCCUPIED (no atomics needed).
        mark_endpoints_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_grid_, d_hits_x_, d_hits_y_,
            num_hits, num_cells_, resolution_,
            origin_x_, origin_y_, grid_size_);

        // Kernel 2: Bresenham ray tracing -- mark free space UNOCCUPIED.
        ray_trace_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_grid_,
            scan.pose.x, scan.pose.y,
            d_hits_x_, d_hits_y_,
            num_hits, num_cells_, resolution_,
            origin_x_, origin_y_, grid_size_);

        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Copy the device grid back to the host (call once after all scans).
void OccupancyGrid::sync_from_device()
{
    size_t grid_bytes = static_cast<size_t>(num_cells_) * num_cells_ * sizeof(int8_t);
    CUDA_CHECK(cudaMemcpy(cells_.data(), d_grid_, grid_bytes,
                          cudaMemcpyDeviceToHost));
}

// Write the grid to a PGM (Portable Gray Map) image file.
// UNKNOWN -> gray (128), UNOCCUPIED -> white (255), OCCUPIED -> black (0).
bool OccupancyGrid::write_pgm(const std::string &filepath) const
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
