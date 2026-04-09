// occupancy_map_cuda.cu
// First-pass CUDA-accelerated occupancy grid mapping.
// Strategy: Bresenham ray tracing stays on the CPU. The GPU handles
// endpoint marking -- one thread per LIDAR hit writes OCCUPIED into the
// grid. This avoids the need for atomics since each hit cell is written
// with the same value (OCCUPIED) so write conflicts are benign.
// After the GPU marks endpoints the CPU does ray tracing on the host
// copy of the grid for the free-space (UNOCCUPIED) cells.
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
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__   \
                      << " -- " << cudaGetErrorString(err) << "\n";        \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Timing Report Helper
// ---------------------------------------------------------------------------
// Prints a label and elapsed time in aligned columns to both an ostream
// and std::cout. Replaces the old snprintf + char buf approach.
static void print_timing_line(std::ostream &rpt,
                              const std::string &label, double ms)
{
    auto flags = std::cout.flags();
    auto prec  = std::cout.precision();

    std::cout << std::left << std::setw(35) << label
              << " " << std::right << std::setw(10)
              << std::fixed << std::setprecision(3) << ms << " ms\n";

    rpt << std::left << std::setw(35) << label
        << " " << std::right << std::setw(10)
        << std::fixed << std::setprecision(3) << ms << " ms\n";

    // Restore original format state so callers are not surprised.
    std::cout.flags(flags);
    std::cout.precision(prec);
}

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

    // Bresenham ray trace from robot position to hit point.
    // Marks all cells along the ray as UNOCCUPIED. Does NOT mark the
    // endpoint -- that is handled by the GPU kernel now.
    //
    // In this first CUDA pass the CPU is responsible for free-space
    // ray tracing because concurrent Bresenham rays can cross the same
    // cells which would require atomics on the GPU. That optimization
    // comes in the next pass.
    void ray_trace_free_space(float ox, float oy, float hx, float hy)
    {
        int c0, r0, c1, r1;
        if (!world_to_grid(ox, oy, c0, r0))
            return;
        if (!world_to_grid(hx, hy, c1, r1))
            return;

        // Bresenham line from (c0,r0) to (c1,r1).
        // Only marks intermediate cells as UNOCCUPIED -- stops before
        // the endpoint so the GPU's OCCUPIED mark is not overwritten.
        bresenham_free_space(cells_.data(), num_cells_, c0, r0, c1, r1);
    }

    // Process an entire LIDAR scan using the hybrid CPU/GPU approach.
    // 1) Launch GPU kernel to mark all hit endpoints as OCCUPIED.
    // 2) Sync the device grid back to host.
    // 3) Run CPU Bresenham for each ray to mark free space.
    // 4) Copy the updated host grid back to device for the next scan.
    void process_scan_cuda(const LidarScan &scan,
                           std::vector<TimingRecord> &timing_log)
    {
        int num_hits = static_cast<int>(scan.hits.size());
        if (num_hits == 0)
            return;

        // -- Pack hit coordinates into flat arrays for the GPU -----------
        std::vector<float> hits_x(num_hits);
        std::vector<float> hits_y(num_hits);
        for (int i = 0; i < num_hits; ++i)
        {
            hits_x[i] = scan.hits[i].first;
            hits_y[i] = scan.hits[i].second;
        }

        // Allocate device hit arrays
        float *d_hits_x = nullptr;
        float *d_hits_y = nullptr;
        size_t hits_bytes = num_hits * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_hits_x, hits_bytes));
        CUDA_CHECK(cudaMalloc(&d_hits_y, hits_bytes));

        // Copy hits to device
        {
            ScopedTimer t("  H2D hit data", timing_log);
            CUDA_CHECK(cudaMemcpy(d_hits_x, hits_x.data(), hits_bytes,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_hits_y, hits_y.data(), hits_bytes,
                                  cudaMemcpyHostToDevice));
        }

        // Launch endpoint-marking kernel
        {
            ScopedTimer t("  GPU endpoint marking", timing_log);
            int num_blocks = (num_hits + BLOCK_SIZE - 1) / BLOCK_SIZE;
            mark_endpoints_kernel<<<num_blocks, BLOCK_SIZE>>>(
                d_grid_, d_hits_x, d_hits_y,
                num_hits, num_cells_, resolution_,
                origin_x_, origin_y_, grid_size_);

            // Sync so the timer captures actual kernel execution time.
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Copy grid back from device so CPU can do ray tracing
        {
            ScopedTimer t("  D2H grid sync", timing_log);
            size_t grid_bytes = static_cast<size_t>(num_cells_) * num_cells_ * sizeof(int8_t);
            CUDA_CHECK(cudaMemcpy(cells_.data(), d_grid_, grid_bytes,
                                  cudaMemcpyDeviceToHost));
        }

        // CPU Bresenham ray tracing for free space
        {
            ScopedTimer t("  CPU ray tracing", timing_log);
            for (const auto &[hx, hy] : scan.hits)
            {
                ray_trace_free_space(scan.pose.x, scan.pose.y, hx, hy);
            }
        }

        // Push updated grid back to device for next scan
        {
            ScopedTimer t("  H2D grid sync", timing_log);
            size_t grid_bytes = static_cast<size_t>(num_cells_) * num_cells_ * sizeof(int8_t);
            CUDA_CHECK(cudaMemcpy(d_grid_, cells_.data(), grid_bytes,
                                  cudaMemcpyHostToDevice));
        }

        // Free per-scan device allocations
        CUDA_CHECK(cudaFree(d_hits_x));
        CUDA_CHECK(cudaFree(d_hits_y));
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

    // Device copy of the grid. Kept in sync with cells_ across scans.
    int8_t *d_grid_;

    // Bresenham (free-space only)

    // Bresenham line rasterization for free space only.
    // Marks every cell along the ray as UNOCCUPIED EXCEPT the final cell
    // (c1, r1) which was already marked OCCUPIED by the GPU kernel.
    //
    // grid       Pointer to the flat grid array.
    // width      Number of cells per row (grid is width x width).
    // c0,r0      Start cell (robot position).
    // c1,r1      End cell (LIDAR hit) -- not modified by this function.
    static void bresenham_free_space(int8_t *grid, int width,
                                     int c0, int r0, int c1, int r1)
    {
        int dc = std::abs(c1 - c0);
        int dr = std::abs(r1 - r0);
        int sc = (c0 < c1) ? 1 : -1;
        int sr = (r0 < r1) ? 1 : -1;
        int err = dc - dr;

        int c = c0, r = r0;

        while (true)
        {
            // Stop before marking the endpoint -- the GPU already wrote
            // OCCUPIED there and we don't want to overwrite it.
            if (c == c1 && r == r1)
                break;

            // All intermediate cells are free space.
            grid[r * width + c] = UNOCCUPIED;

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
    }
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
        std::ofstream rpt("timing_report.txt");
        rpt << "=== Occupancy Map CUDA (Pass 1) Timing Report ===\n\n";
        rpt << "Grid: " << grid.num_cells() << " x " << grid.num_cells()
            << "  resolution=" << grid.resolution() << " m\n";
        rpt << "Scans processed: " << scans.size() << "\n\n";

        // Accumulate per-stage totals across all scans for the breakdown.
        double total_scan_ms = 0.0;
        double total_h2d_hits_ms = 0.0;
        double total_gpu_mark_ms = 0.0;
        double total_d2h_grid_ms = 0.0;
        double total_cpu_ray_ms = 0.0;
        double total_h2d_grid_ms = 0.0;

        for (const auto &r : timing_log)
        {
            if (r.label.rfind("Scan #", 0) == 0)
                total_scan_ms += r.elapsed_ms;
            else if (r.label == "  H2D hit data")
                total_h2d_hits_ms += r.elapsed_ms;
            else if (r.label == "  GPU endpoint marking")
                total_gpu_mark_ms += r.elapsed_ms;
            else if (r.label == "  D2H grid sync")
                total_d2h_grid_ms += r.elapsed_ms;
            else if (r.label == "  CPU ray tracing")
                total_cpu_ray_ms += r.elapsed_ms;
            else if (r.label == "  H2D grid sync")
                total_h2d_grid_ms += r.elapsed_ms;
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

        // Print the per-stage breakdown totals.
        std::cout << "\n--- Per-stage totals (summed across all scans) ---\n";
        rpt << "\n--- Per-stage totals (summed across all scans) ---\n";

        print_timing_line(rpt, "H2D hit data (total)", total_h2d_hits_ms);
        print_timing_line(rpt, "GPU endpoint marking (total)", total_gpu_mark_ms);
        print_timing_line(rpt, "D2H grid sync (total)", total_d2h_grid_ms);
        print_timing_line(rpt, "CPU ray tracing (total)", total_cpu_ray_ms);
        print_timing_line(rpt, "H2D grid sync (total)", total_h2d_grid_ms);

        // Per-scan average.
        if (!scans.empty())
        {
            double avg = total_scan_ms / scans.size();
            std::cout << "\n";
            rpt << "\n";
            print_timing_line(rpt, "Average per scan", avg);
        }

        std::cout << "\nFull timing log: timing_report.txt\n";
    }

    return 0;
}
