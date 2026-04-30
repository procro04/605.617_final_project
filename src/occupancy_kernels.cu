#include "occupancy_kernels.cuh"
#include "Common.h"
#include "CudaHelpers.h"

// ---------------------------------------------------------------------------
// Device helper: atomic write to an int8_t cell using 4-byte atomicCAS.
// ---------------------------------------------------------------------------
// int8_t has no native CUDA atomic, so we operate on the 4-byte word that
// contains the target byte.  The cell is updated from UNKNOWN to `value`
// only; if the cell is already OCCUPIED or UNOCCUPIED the write is skipped.
// cudaMalloc guarantees at least 256-byte alignment, so the int* cast is safe.
inline __device__ static void atomic_set_if_unknown(int8_t *d_grid, int idx, int8_t value)
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
