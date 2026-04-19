// combine_intel_data.cpp
// Reads separate ODO and LASER files from the Intel Research Lab dataset
// and writes a single combined file in the format produced by
// generate_test_data.cpp (one line per scan):
//
//   x_robot  y_robot  theta  hx1 hy1  hx2 hy2  ...  hxN hyN
//
// Input files (one data line per scan; lines beginning with '#' are ignored):
//   odo.txt   -- x  y  theta
//   laser.txt -- hx1 hy1  hx2 hy2  ...  hxN hyN   (world-frame hit coords)
//
// Usage:
//   ./combine_intel_data <odo.txt> <laser.txt> [output.txt]
//   If output.txt is omitted the combined data is written to stdout.

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Common.h"

void print_usage()
{
    std::cout << "Provide the ODO and LASER .txt files to combine.\n"
              << "./combine_intel_data <ODO.txt> <LASER.txt> [output.txt]\n";
}

// Read all non-empty, non-comment lines from a file into a vector.
static std::vector<std::string> read_data_lines(const std::string &path)
{
    std::vector<std::string> lines;
    std::ifstream f(path);
    if (!f.is_open())
    {
        std::cerr << "ERROR: Cannot open " << path << "\n";
        return lines;
    }
    std::string line;
    while (std::getline(f, line))
    {
        if (!line.empty() && line[0] != '#')
            lines.push_back(std::move(line));
    }
    return lines;
}

// Load paired ODO + LASER files into LidarScan structs.
// Each ODO line:   x  y  theta
// Each LASER line: hx1 hy1  hx2 hy2  ...  hxN hyN   (world-frame hit coords)
// Lines are matched positionally: ODO line i pairs with LASER line i.
std::vector<LidarScan> load_lidar_dataset(
    const std::string &odo_path,
    const std::string &laser_path)
{
    std::vector<std::string> odo_lines   = read_data_lines(odo_path);
    std::vector<std::string> laser_lines = read_data_lines(laser_path);

    if (odo_lines.empty() || laser_lines.empty())
        return {};

    if (odo_lines.size() != laser_lines.size())
    {
        std::cerr << "WARNING: ODO has " << odo_lines.size()
                  << " scans but LASER has " << laser_lines.size()
                  << " scans. Using the smaller count.\n";
    }

    size_t num_scans = std::min(odo_lines.size(), laser_lines.size());
    std::vector<LidarScan> scans;
    scans.reserve(num_scans);

    for (size_t i = 0; i < num_scans; ++i)
    {
        std::istringstream odo_iss(odo_lines[i]);
        LidarScan scan;
        if (!(odo_iss >> scan.pose.x >> scan.pose.y >> scan.pose.theta))
        {
            std::cerr << "WARNING: Could not parse ODO line " << i + 1 << " -- skipping.\n";
            continue;
        }

        std::istringstream laser_iss(laser_lines[i]);
        float hx, hy;
        while (laser_iss >> hx >> hy)
            scan.hits.emplace_back(hx, hy);

        if (!scan.hits.empty())
            scans.push_back(std::move(scan));
    }

    std::cerr << "Loaded " << scans.size() << " scans from "
              << odo_path << " + " << laser_path << "\n";
    return scans;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        print_usage();
        return 1;
    }

    std::vector<LidarScan> scans = load_lidar_dataset(argv[1], argv[2]);
    if (scans.empty())
    {
        std::cerr << "No scans loaded. Exiting.\n";
        return 1;
    }

    // Write to a file if provided, otherwise stdout.
    std::ostream *out = &std::cout;
    std::ofstream outfile;
    if (argc >= 4)
    {
        outfile.open(argv[3]);
        if (!outfile.is_open())
        {
            std::cerr << "ERROR: Cannot open output file " << argv[3] << "\n";
            return 1;
        }
        out = &outfile;
    }

    for (const auto &scan : scans)
    {
        *out << scan.pose.x << " " << scan.pose.y << " " << scan.pose.theta;
        for (const auto &[hx, hy] : scan.hits)
            *out << " " << hx << " " << hy;
        *out << "\n";
    }

    std::cerr << "Wrote " << scans.size() << " scans.\n";
    return 0;
}
