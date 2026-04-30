#pragma once

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
                                      float grid_size);

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
                                 float grid_size);
