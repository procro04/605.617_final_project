/**
 * generate_test_data.cpp
 *
 * Generates synthetic LIDAR scan data for testing the occupancy map pipeline.
 * Simulates a robot at the origin surrounded by rectangular walls, rotating
 * a 180-degree LIDAR sweep.
 *
 * Output format (one scan per line):
 *   x_robot  y_robot  theta  x1 y1  x2 y2  ...  xN yN
 *
 * Build:  g++ -O2 -std=c++17 -o generate_test_data generate_test_data.cpp
 * Usage:  ./generate_test_data > test_lidar.txt
 */

#include <cmath>
#include <cstdio>
#include <vector>

struct Wall {
    float x0, y0, x1, y1;  // line segment endpoints
};

/// Ray-segment intersection. Returns distance t along the ray, or -1.
float ray_segment_intersect(float ox, float oy, float dx, float dy,
                            float sx0, float sy0, float sx1, float sy1) {
    float ex = sx1 - sx0, ey = sy1 - sy0;
    float denom = dx * ey - dy * ex;
    if (std::abs(denom) < 1e-8f) return -1.0f;

    float t = ((sx0 - ox) * ey - (sy0 - oy) * ex) / denom;
    float u = ((sx0 - ox) * dy - (sy0 - oy) * dx) / denom;

    if (t > 0.0f && u >= 0.0f && u <= 1.0f) return t;
    return -1.0f;
}

int main() {
    // Define a rectangular room: 10m x 8m centered at origin
    std::vector<Wall> walls = {
        { -5.0f, -4.0f,  5.0f, -4.0f },  // bottom
        {  5.0f, -4.0f,  5.0f,  4.0f },  // right
        {  5.0f,  4.0f, -5.0f,  4.0f },  // top
        { -5.0f,  4.0f, -5.0f, -4.0f },  // left
        // Internal obstacle: small box
        {  1.0f,  0.0f,  2.0f,  0.0f },
        {  2.0f,  0.0f,  2.0f,  1.5f },
        {  2.0f,  1.5f,  1.0f,  1.5f },
        {  1.0f,  1.5f,  1.0f,  0.0f },
    };

    const float max_range = 15.0f;
    const int   num_beams = 180;          // beams per scan
    const int   num_scans = 50;           // robot takes 50 scans as it moves

    for (int s = 0; s < num_scans; ++s) {
        // Robot moves in a small circle
        float frac = static_cast<float>(s) / num_scans;
        float rx = 1.5f * std::cos(frac * 2.0f * M_PI) - 1.0f;
        float ry = 1.0f * std::sin(frac * 2.0f * M_PI);
        float rtheta = frac * 2.0f * M_PI;

        std::printf("%.4f %.4f %.4f", rx, ry, rtheta);

        for (int b = 0; b < num_beams; ++b) {
            float angle = rtheta + (static_cast<float>(b) / num_beams - 0.5f) * M_PI;
            float dx = std::cos(angle);
            float dy = std::sin(angle);

            float best_t = max_range;
            for (const auto& w : walls) {
                float t = ray_segment_intersect(rx, ry, dx, dy,
                                                w.x0, w.y0, w.x1, w.y1);
                if (t > 0.0f && t < best_t) best_t = t;
            }

            float hx = rx + dx * best_t;
            float hy = ry + dy * best_t;
            std::printf(" %.4f %.4f", hx, hy);
        }

        std::printf("\n");
    }

    return 0;
}
