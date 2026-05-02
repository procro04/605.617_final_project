// Generates high-density synthetic LIDAR scan data for GPU performance testing.
// Fires 1800 beams per scan (360-degree sweep at 0.2 deg/beam), giving enough
// parallel work to fill multiple CUDA thread blocks and better stress the GPU path.
//
// Output format (one scan per line):
//   x_robot  y_robot  theta  hx1 hy1  hx2 hy2  ...  hxN hyN
//
// Usage:  ./generate_test_data_hd > test_lidar_hd.txt

#include <cmath>
#include <iostream>
#include <vector>

struct Wall
{
    float x0, y0, x1, y1;
};

float ray_segment_intersect(float ox, float oy, float dx, float dy,
                            float sx0, float sy0, float sx1, float sy1)
{
    float ex = sx1 - sx0, ey = sy1 - sy0;
    float denom = dx * ey - dy * ex;
    if (std::abs(denom) < 1e-8f)
        return -1.0f;

    float t = ((sx0 - ox) * ey - (sy0 - oy) * ex) / denom;
    float u = ((sx0 - ox) * dy - (sy0 - oy) * dx) / denom;

    if (t > 0.0f && u >= 0.0f && u <= 1.0f)
        return t;
    return -1.0f;
}

int main()
{
    // Larger room: 20m x 15m with several internal obstacles.
    std::vector<Wall> walls = {
        // Outer boundary
        {-10.f, -7.5f,  10.f, -7.5f},  // bottom
        { 10.f, -7.5f,  10.f,  7.5f},  // right
        { 10.f,  7.5f, -10.f,  7.5f},  // top
        {-10.f,  7.5f, -10.f, -7.5f},  // left

        // Central dividing wall with a gap
        {-1.f, -7.5f, -1.f, -1.f},
        {-1.f,  1.f,  -1.f,  7.5f},

        // Box obstacle (left side)
        {-7.f, -4.f, -4.f, -4.f},
        {-4.f, -4.f, -4.f, -1.f},
        {-4.f, -1.f, -7.f, -1.f},
        {-7.f, -1.f, -7.f, -4.f},

        // Box obstacle (right side)
        {4.f, 1.f, 7.f, 1.f},
        {7.f, 1.f, 7.f, 5.f},
        {7.f, 5.f, 4.f, 5.f},
        {4.f, 5.f, 4.f, 1.f},

        // Diagonal wall (right-bottom)
        {3.f, -7.5f, 10.f, -2.f},
    };

    const float max_range = 30.f;
    // const int num_beams = 16384;  // stress-test value: 16K beams fills ~64 CUDA blocks
    // const int num_beams = 40000;  // stress-test value: 16K beams fills ~64 CUDA blocks
    const int num_beams = 100000;  // stress-test value: 16K beams fills ~64 CUDA blocks
    const int num_scans = 100;

    for (int s = 0; s < num_scans; ++s)
    {
        // Robot traces a figure-8 path through the environment
        float frac = static_cast<float>(s) / num_scans;
        float t = frac * 2.0f * static_cast<float>(M_PI);
        float rx = 3.5f * std::sin(t);
        float ry = 2.0f * std::sin(t) * std::cos(t);  // lemniscate y
        float rtheta = t;

        std::cout << rx << " " << ry << " " << rtheta;

        for (int b = 0; b < num_beams; ++b)
        {
            // Full 360-degree sweep
            float angle = static_cast<float>(b) / num_beams * 2.0f * static_cast<float>(M_PI);
            float dx = std::cos(angle);
            float dy = std::sin(angle);

            float best_t = max_range;
            for (const auto &w : walls)
            {
                float hit = ray_segment_intersect(rx, ry, dx, dy,
                                                  w.x0, w.y0, w.x1, w.y1);
                if (hit > 0.0f && hit < best_t)
                    best_t = hit;
            }

            std::cout << " " << (rx + dx * best_t) << " " << (ry + dy * best_t);
        }
        std::cout << "\n";
    }

    return 0;
}
