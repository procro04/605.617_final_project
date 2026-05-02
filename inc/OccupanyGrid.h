#pragma once

#include "Common.h"
#include "CodeTimer.h"
#include "CudaHelpers.h"

std::vector<LidarScan> load_lidar_dataset(const std::string &filepath);

// ---------------------------------------------------------------------------
// Occupancy Grid  (flat, CUDA-accelerated)
// ---------------------------------------------------------------------------
class OccupancyGrid
{
public:
    // grid_size_meters  Physical side length of the square grid.
    // resolution        Meters per grid cell.
    // origin_x/y        World coordinate of the grid center.
    OccupancyGrid(float grid_size_meters, float resolution,
                  float origin_x = 0.0f, float origin_y = 0.0f);
    ~OccupancyGrid();

    // Simple getters can stay in the header
    int num_cells() const { return num_cells_; }
    float resolution() const { return resolution_; }
    float grid_size() const { return grid_size_; }
    int8_t *data() { return cells_.data(); }
    const int8_t *data() const { return cells_.data(); }
    // Flat index from (col, row).
    inline int idx(int col, int row) const { return row * num_cells_ + col; }

    // Convert world (x, y) -> grid (col, row). Returns false if out of bounds.
    bool world_to_grid(float wx, float wy, int &col, int &row) const;
    // Set a cell's state (bounds-checked).
    void set_cell(int col, int row, CellState state);
    CellState get_cell(int col, int row) const;
    // Mark a single cell as OCCUPIED (the endpoint of a ray).
    void point_update(float wx, float wy, CellState state);

    // Process an entire LIDAR scan entirely on the GPU.
    // 1) Copy hit coordinates to the device.
    // 2) Kernel 1: mark all endpoints OCCUPIED (no atomics needed).
    // 3) Kernel 2: Bresenham ray trace for free space using atomicCAS.
    // The device grid is NOT synced back to the host between scans; the
    // host copy is only updated when sync_from_device() is called.
    void process_scan_cuda(const LidarScan &scan,
                           std::vector<TimingRecord> &timing_log);
    // Copy the device grid back to the host (call once after all scans).
    void sync_from_device();
    // Write the grid to a PGM (Portable Gray Map) image file.
    // UNKNOWN -> gray (128), UNOCCUPIED -> white (255), OCCUPIED -> black (0).
    bool write_pgm(const std::string &filepath) const;

private:
    float grid_size_;
    float resolution_;
    float origin_x_;
    float origin_y_;
    int num_cells_;

    // Flat, contiguous cell storage -- row-major. Host copy.
    std::vector<int8_t> cells_;

    // Device copy of the grid. Persists across scans; synced to host only
    // on demand via sync_from_device().
    int8_t *d_grid_;

    // Pre-allocated device hit buffers. Grown on demand; never shrunk.
    // Eliminates cudaMalloc/cudaFree overhead on every scan.
    float *d_hits_x_;
    float *d_hits_y_;
    int    hits_capacity_;
};
