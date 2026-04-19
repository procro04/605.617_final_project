
// Function to provide intel data in a single file rather than separate.
// The test code I have already writes a single file so I don't want to
// modify what is working. Would rather add this helper

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "Common.h"

void print_usage()
{
    std::cout << "Provide the ODO and LASER .txt files to combine.\n"
              << "./combine_intel_data <ODO.txt> <LASER.txt>"
              << std::endl;
}

std::vector<LidarScan> load_lidar_dataset(
    const std::string &odo_path,
    const std::string &laser_path)
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

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        print_usage();
        return 1;
    }


}
