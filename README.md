# 605.617_final_project
Final Project For GPU Programming Course. GPU accelerated occupancy map.

# Folder Layout
- `inc/` - include files
- `reference_code/` - my reference python code from my robotics class. These files
provide the base of the algorithm to convert to C++/CUDA.
- `src/` - cpp source files
- `scripts/` - any useful scripts for debugging, visualization, etc

# Build
- CMakeLists.txt should do everything. You may need to find and specify your
particular platform. My GPU works with 75.
```bash
mkdir build
cd build
cmake ..
cmake --build .
```
- That will give you 3 executables:
  - generate_test_data
  - occupancy_map_cpu
  - occupany_map_cuda

# Getting Data
- I plan to use the intel set here: https://github.com/1988kramer/intel_dataset/tree/master/data
eventually but have started smaller with a simple c++ script for an easy test case
- Run `./generate_test_data > <your_file_name.txt>` to generate some simple input data

# Running Code
- I have 2 instances of the code right now. `occupancy_map_cpu.cpp` and
`occupancy_map_cuda.cu`. I plan to integrate these better so you can run
each version from the same main file for easy speed comparisons however they are
separate right now (with a lot of duplicate code)
- Run: `./occupany_map_cpu <your_test_data.txt>` and it will run the CPU version
of the code and output a .pgm file for the generated map.
- Run: `./occupany_map_cuda <your_test_data.txt>` and it will run the GPU version
of the code and output a .pgm file for the generated map.

# Getting Results
- There is a python script `visualize_map.py` I am using to see if the generated
.pgm file looks anything like the input data.
- To view run: `python3 visualize_map.py <your_test_data.txt> <your_pgm_out.pgm>`
