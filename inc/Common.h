#pragma once

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Configuration & Constants
// ---------------------------------------------------------------------------

// Cell states -- matches Python MapType. Using int8_t so the grid is compact.
// These values also double as PGM pixel intensities after remapping.
enum CellState : int8_t
{
    UNKNOWN = -1,
    UNOCCUPIED = 0,
    OCCUPIED = 1
};

// Block size for the endpoint-marking kernel. Can play around with values but
// 256 is reasonable
static const int BLOCK_SIZE = 256;

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
