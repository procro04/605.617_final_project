// occupancy_map_cuda.cu
// CUDA-accelerated occupancy grid mapping with full GPU ray tracing.
// Strategy: Two back-to-back GPU kernels per scan.
//   1) mark_endpoints_kernel  -- one thread per hit writes OCCUPIED.
//      Concurrent same-value writes are benign; no atomics needed.
//   2) ray_trace_kernel       -- one thread per hit runs Bresenham and
//      writes UNOCCUPIED along the ray.  Concurrent rays can share cells,
//      so we use a 4-byte atomicCAS trick to update int8_t cells safely:
//      a cell is only changed from UNKNOWN to UNOCCUPIED, never from
//      OCCUPIED to UNOCCUPIED, so previously-marked endpoints are preserved.
//
// Usage:
//   ./occupancy_map_cuda <lidar_data.txt> [grid_size] [resolution]
//   lidar_data.txt  -- Intel Research Lab format: each line is
//                    "x_robot y_robot theta  x1 y1  x2 y2 ... xN yN"
//                    (the first 3 values are the robot pose, followed by
//                    pairs of LIDAR hit coordinates in world frame)
//   grid_size       -- physical side length in meters (default: 40.0)
//   resolution      -- meters per cell (default: 0.05)
// Outputs:
//   occupancy_####.pgm  -- grayscale image per scan (or final only, configurable)
//   timing_report.txt   -- per-stage benchmark results

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "CodeTimer.h"
#include "Common.h"

// ---------------------------------------------------------------------------
// CUDA Error Checking
// ---------------------------------------------------------------------------

// Wrap every CUDA API call with this to catch errors early.
#define CUDA_CHECK(call)                                                 \
    do                                                                   \
    {                                                                    \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess)                                          \
        {                                                                \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " -- " << cudaGetErrorString(err) << "\n";      \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

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

// ---------------------------------------------------------------------------
// Device helper: atomic write to an int8_t cell using 4-byte atomicCAS.
// ---------------------------------------------------------------------------
// int8_t has no native CUDA atomic, so we operate on the 4-byte word that
// contains the target byte.  The cell is updated from UNKNOWN to `value`
// only; if the cell is already OCCUPIED or UNOCCUPIED the write is skipped.
// cudaMalloc guarantees at least 256-byte alignment, so the int* cast is safe.
__device__ static void atomic_set_if_unknown(int8_t *d_grid, int idx, int8_t value)
{
    int byte_lane = idx & 3;
    unsigned int shift = static_cast<unsigned int>(byte_lane) * 8u;
    unsigned int byte_mask = 0xFFu << shift;
    unsigned int *word_ptr = reinterpret_cast<unsigned int *>(d_grid) + (idx >> 2);

    unsigned int old_word = *word_ptr;
    while (true)
    {
        // Check whether the target byte is still UNKNOWN (0xFF when unsigned).
        uint8_t cur = static_cast<uint8_t>((old_word >> shift) & 0xFFu);
        if (cur != static_cast<uint8_t>(UNKNOWN))
            break;

        unsigned int new_word =
            (old_word & ~byte_mask) |
            (static_cast<unsigned int>(static_cast<uint8_t>(value)) << shift);
        unsigned int prev = atomicCAS(word_ptr, old_word, new_word);
        if (prev == old_word)
            break;       // success
        old_word = prev; // retry with the freshly read word
    }
}

// ---------------------------------------------------------------------------
// GPU Kernel: Endpoint Marking
// ---------------------------------------------------------------------------
// Each thread handles one LIDAR hit. It converts the world-frame (x,y)
// coordinate to a grid cell and writes OCCUPIED into the flat grid array.
// Since every thread writes the same value (OCCUPIED=1) there is no
// data race -- concurrent writes of the same byte value are safe on CUDA
// even without atomics.
//
// d_grid       -- device pointer to the flat grid (num_cells * num_cells)
// d_hits_x     -- device array of hit x-coordinates (world frame)
// d_hits_y     -- device array of hit y-coordinates (world frame)
// num_hits     -- number of hits in this batch
// num_cells    -- grid dimension (grid is num_cells x num_cells)
// resolution   -- meters per cell
// origin_x     -- world x of the grid center
// origin_y     -- world y of the grid center
// grid_size    -- physical side length in meters
__global__ void mark_endpoints_kernel(int8_t *d_grid,
                                      const float *d_hits_x,
                                      const float *d_hits_y,
                                      int num_hits,
                                      int num_cells,
                                      float resolution,
                                      float origin_x,
                                      float origin_y,
                                      float grid_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hits)
        return;

    // World-to-grid conversion, same math as the CPU version.
    float local_x = d_hits_x[tid] - (origin_x - grid_size / 2.0f);
    float local_y = d_hits_y[tid] - (origin_y - grid_size / 2.0f);
    int col = (int)floorf(local_x / resolution);
    int row = (int)floorf(local_y / resolution);

    // Bounds check -- silently skip out-of-range hits.
    if (col >= 0 && col < num_cells && row >= 0 && row < num_cells)
    {
        d_grid[row * num_cells + col] = OCCUPIED;
    }
}

// ---------------------------------------------------------------------------
// GPU Kernel: Bresenham Ray Tracing (free space)
// ---------------------------------------------------------------------------
// One thread per LIDAR hit.  Each thread walks the Bresenham line from the
// robot position to (but not including) the hit endpoint and marks every
// intermediate cell UNOCCUPIED using the atomic helper above.
// Cells already set to OCCUPIED (by mark_endpoints_kernel or a prior scan)
// are preserved because atomic_set_if_unknown only updates UNKNOWN cells.
//
// robot_x/y    -- world-frame robot position (scalar, shared by all rays)
// d_hits_x/y   -- per-ray hit positions (world frame)
// Uses code conventions from the wikipedia page:
// https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
__global__ void ray_trace_kernel(int8_t *d_grid,
                                 float robot_x,
                                 float robot_y,
                                 const float *d_hits_x,
                                 const float *d_hits_y,
                                 int num_hits,
                                 int num_cells,
                                 float resolution,
                                 float origin_x,
                                 float origin_y,
                                 float grid_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hits)
        return;

    float off_x = origin_x - grid_size / 2.0f;
    float off_y = origin_y - grid_size / 2.0f;

    // Start and end grid coordinates. column/row
    int c0 = (int)floorf((robot_x - off_x) / resolution);
    int r0 = (int)floorf((robot_y - off_y) / resolution);
    int c1 = (int)floorf((d_hits_x[tid] - off_x) / resolution);
    int r1 = (int)floorf((d_hits_y[tid] - off_y) / resolution);

    // Skip rays whose origin or endpoint falls outside the grid.
    if (c0 < 0 || c0 >= num_cells || r0 < 0 || r0 >= num_cells)
        return;
    if (c1 < 0 || c1 >= num_cells || r1 < 0 || r1 >= num_cells)
        return;

    // delta column/delta row
    int dc = abs(c1 - c0);
    int dr = abs(r1 - r0);
    int sc = (c0 < c1) ? 1 : -1;
    int sr = (r0 < r1) ? 1 : -1;
    int err = dc - dr;
    int c = c0, r = r0;

    while (!(c == c1 && r == r1))
    {
        // Atomically mark the cell UNOCCUPIED only if it is still UNKNOWN.
        // This preserves OCCUPIED endpoints from the first kernel and from
        // previous scans.
        atomic_set_if_unknown(d_grid, r * num_cells + c, (int8_t)UNOCCUPIED);

        int e2 = 2 * err;
        if (e2 > -dr)
        {
            err -= dr;
            c += sc;
        }
        if (e2 < dc)
        {
            err += dc;
            r += sr;
        }
    }
    // (c1, r1) is already OCCUPIED -- do not touch it.
}

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
                  float origin_x = 0.0f, float origin_y = 0.0f)
        : grid_size_(grid_size_meters),
          resolution_(resolution),
          origin_x_(origin_x),
          origin_y_(origin_y),
          d_grid_(nullptr)
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

    ~OccupancyGrid()
    {
        if (d_grid_)
        {
            cudaFree(d_grid_);
            d_grid_ = nullptr;
        }
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

    // Mark a single cell as OCCUPIED (the endpoint of a ray).
    void point_update(float wx, float wy, CellState state)
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
    void process_scan_cuda(const LidarScan &scan,
                           std::vector<TimingRecord> &timing_log)
    {
        int num_hits = static_cast<int>(scan.hits.size());
        if (num_hits == 0)
            return;

        // Pack hit coordinates into flat arrays for the GPU.
        std::vector<float> hits_x(num_hits);
        std::vector<float> hits_y(num_hits);
        for (int i = 0; i < num_hits; ++i)
        {
            hits_x[i] = scan.hits[i].first;
            hits_y[i] = scan.hits[i].second;
        }

        // Allocate device hit arrays.
        float *d_hits_x = nullptr;
        float *d_hits_y = nullptr;
        size_t hits_bytes = num_hits * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_hits_x, hits_bytes));
        CUDA_CHECK(cudaMalloc(&d_hits_y, hits_bytes));

        // Copy hits to device.
        {
            ScopedTimer t("  H2D hit data", timing_log);
            CUDA_CHECK(cudaMemcpy(d_hits_x, hits_x.data(), hits_bytes,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_hits_y, hits_y.data(), hits_bytes,
                                  cudaMemcpyHostToDevice));
        }

        int num_blocks = (num_hits + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Kernel 1: mark endpoints OCCUPIED.
        {
            ScopedTimer t("  GPU endpoint marking", timing_log);
            mark_endpoints_kernel<<<num_blocks, BLOCK_SIZE>>>(
                d_grid_, d_hits_x, d_hits_y,
                num_hits, num_cells_, resolution_,
                origin_x_, origin_y_, grid_size_);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Kernel 2: Bresenham ray tracing -- mark free space UNOCCUPIED.
        // Uses atomicCAS (via atomic_set_if_unknown) so concurrent rays that
        // cross the same cell are handled safely, and OCCUPIED endpoints are
        // never overwritten.
        {
            ScopedTimer t("  GPU ray tracing", timing_log);
            ray_trace_kernel<<<num_blocks, BLOCK_SIZE>>>(
                d_grid_,
                scan.pose.x, scan.pose.y,
                d_hits_x, d_hits_y,
                num_hits, num_cells_, resolution_,
                origin_x_, origin_y_, grid_size_);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Free per-scan device allocations.
        CUDA_CHECK(cudaFree(d_hits_x));
        CUDA_CHECK(cudaFree(d_hits_y));
    }

    // Copy the device grid back to the host (call once after all scans).
    void sync_from_device()
    {
        size_t grid_bytes = static_cast<size_t>(num_cells_) * num_cells_ * sizeof(int8_t);
        CUDA_CHECK(cudaMemcpy(cells_.data(), d_grid_, grid_bytes,
                              cudaMemcpyDeviceToHost));
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

    // Flat, contiguous cell storage -- row-major. Host copy.
    std::vector<int8_t> cells_;

    // Device copy of the grid. Persists across scans; synced to host only
    // on demand via sync_from_device().
    int8_t *d_grid_;
};

void print_cuda_device_info()
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found!\n";
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA Device: " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  SMs: " << prop.multiProcessorCount
              << "  Max threads/block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Global memory: " << (prop.totalGlobalMem >> 20) << " MB\n";
}

void print_usage(const char *prog)
{
    std::cout << "Usage: " << prog
              << " <lidar_data.txt> [grid_size_m] [resolution_m]\n"
              << "  grid_size_m   -- physical side length (default 40.0)\n"
              << "  resolution_m  -- meters per cell      (default 0.05)\n";
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::string data_file = argv[1];
    float grid_size = (argc > 2) ? std::stof(argv[2]) : 40.0f;
    float resolution = (argc > 3) ? std::stof(argv[3]) : 0.05f;

    std::vector<TimingRecord> timing_log;

    print_cuda_device_info();

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

    // Compute a reasonable grid origin from the data
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

    // Initialize grid (also allocates device memory
    OccupancyGrid grid(grid_size, resolution, cx, cy);

    // Process all scans
    {
        ScopedTimer t("Process all scans (total)", timing_log);

        for (size_t i = 0; i < scans.size(); ++i)
        {
            ScopedTimer ts("Scan #" + std::to_string(i), timing_log);
            grid.process_scan_cuda(scans[i], timing_log);
        }
    }

    // Sync device grid to host once -- all scans are complete.
    {
        ScopedTimer t("D2H final grid sync", timing_log);
        grid.sync_from_device();
    }

    // Write final output
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
    {
        std::ofstream timing_report("timing_report.txt");
        timing_report << "=== Occupancy Map CUDA (Full GPU Ray Tracing) Timing Report ===\n\n";
        timing_report << "Grid: " << grid.num_cells() << " x " << grid.num_cells()
            << "  resolution=" << grid.resolution() << " m\n";
        timing_report << "Scans processed: " << scans.size() << "\n\n";

        // Accumulate per-stage totals across all scans for the breakdown.
        double total_scan_ms = 0.0;
        double total_h2d_hits_ms = 0.0;
        double total_gpu_mark_ms = 0.0;
        double total_gpu_ray_ms = 0.0;

        for (const auto &r : timing_log)
        {
            if (r.label.rfind("Scan #", 0) == 0)
                total_scan_ms += r.elapsed_ms;
            else if (r.label == "  H2D hit data")
                total_h2d_hits_ms += r.elapsed_ms;
            else if (r.label == "  GPU endpoint marking")
                total_gpu_mark_ms += r.elapsed_ms;
            else if (r.label == "  GPU ray tracing")
                total_gpu_ray_ms += r.elapsed_ms;
        }

        // Print high-level timings first.
        for (const auto &r : timing_log)
        {
            // Only print aggregate records and non-per-scan, non-substep records.
            if (r.label.rfind("Scan #", 0) != 0 &&
                r.label.rfind("  ", 0) != 0)
            {
                print_timing_line(timing_report, r.label, r.elapsed_ms);
            }
        }

        // Print the per-stage breakdown totals.
        std::cout << "\n--- Per-stage totals (summed across all scans) ---\n";
        timing_report << "\n--- Per-stage totals (summed across all scans) ---\n";

        print_timing_line(timing_report, "H2D hit data (total)", total_h2d_hits_ms);
        print_timing_line(timing_report, "GPU endpoint marking (total)", total_gpu_mark_ms);
        print_timing_line(timing_report, "GPU ray tracing (total)", total_gpu_ray_ms);

        // Per-scan average.
        if (!scans.empty())
        {
            double avg = total_scan_ms / scans.size();
            std::cout << "\n";
            timing_report << "\n";
            print_timing_line(timing_report, "Average per scan", avg);
        }

        std::cout << "\nFull timing log: timing_report.txt\n";
    }

    return 0;
}
