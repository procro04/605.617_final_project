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
#include "OccupanyGrid.h"
#include "CudaHelpers.h"

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
