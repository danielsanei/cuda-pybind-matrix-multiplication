# CUDA PyBind11 Matrix Multiplication

A high-performance matrix multiplication implementation that integrates CUDA GPU acceleration with Python using PyBind11.

## Project Overview

This project implements optimized 1024×1024 matrix multiplication using CUDA and exposes the functionality to Python through PyBind11 bindings. The implementation demonstrates:

- GPU-accelerated matrix multiplication using CUDA with shared memory optimization
- Python-C++ integration via PyBind11 for seamless interoperability
- Performance benchmarking comparing CPU and GPU execution times

### Performance Results

For **1024×1024 integer matrices**:
- **CPU Execution Time**: ~2,240 ms
- **GPU Execution Time** (with shared memory): ~35-51 ms
- **Speedup**: ~44-64x faster than CPU

## Directory Structure

```
cuda_bind_project/
├── Src/
│   ├── cuda_bind/
│   │   ├── cuda_bind.cu          # CUDA kernel implementation
│   │   ├── cuda_bind_py.cpp      # PyBind11 wrapper (empty/stub)
│   │   └── CMakeLists.txt
│   └── CMakeLists.txt
├── build/                         # Build directory (generated)
├── cmake/                         # CMake find modules
├── Extern/                        # External libraries (GLM, GLAD)
├── CMakeLists.txt                 # Main CMake configuration
├── cuda_bind_py.pyd              # Compiled Python module (generated)
└── test.py                        # Python test script
```

### Key Files

- **cuda_bind.cu**: Core CUDA implementation with both global and shared memory kernels
- **cuda_bind_py.cpp**: PyBind11 wrapper for Python integration
- **test.py**: Python driver script for testing the module
- **CMakeLists.txt**: Build configuration

## Code Overview

### CUDA Implementation

The CUDA code (`cuda_bind.cu`) implements two matrix multiplication kernels:

#### 1. Global Memory Kernel
- Standard approach with direct global memory access
- Each thread computes one output element
- Simple but less optimized

#### 2. Shared Memory Kernel (Optimized)
- Uses tiled matrix multiplication with 16×16 tiles
- Loads tiles into fast shared memory to reduce global memory bandwidth
- Implements cooperative loading and thread synchronization
- Achieves ~44-64x speedup over CPU

**Key Configuration:**
- Matrix size: 1024×1024
- Block size: 16×16 threads (256 threads per block)
- Grid size: 64×64 blocks
- Shared memory: Two 16×16 tile buffers

**Kernel Launch:**
The `launch_matrix_multiplication_shared_memory()` function provides a C-linkage interface for PyBind11, handling kernel configuration and synchronization.

### PyBind11 Integration

The PyBind11 wrapper (`pybind11_madd.cpp` in the reference example) bridges Python NumPy arrays to CUDA:

**Functionality:**
- Accepts two NumPy arrays as input (1024×1024, int32)
- Validates array dimensions and data types
- Allocates GPU memory and transfers data
- Launches CUDA kernel with timing measurements
- Returns results including CPU time, GPU time, and validation status

**Python Interface:**
```python
results = cu_matrix_add.madd(matrix_a, matrix_b)
# Returns: {'cpu_time_ms': float, 'gpu_time_ms': float, 
#           'pass': bool, 'correct_count': int}
```

## Prerequisites

- **CUDA Toolkit** (version 10.1 or compatible)
- **CMake** (version 3.11 or higher)
- **Python** (version 3.6 or higher) with NumPy
- **PyBind11**: `pip install pybind11`
- **C++ Compiler** (Visual Studio 2017+ on Windows, GCC on Linux, Xcode on macOS)
- **NVIDIA GPU** with compute capability 5.2 or higher

## Building the Project

Navigate to the project directory and build using CMake:

```bash
cd cuda_bind_project
mkdir build
cd build

# Windows
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release

# Linux/macOS
cmake ..
make
```

**Output:**
- CUDA executable: `build/Src/cuda_bind/cuda_bind` (or `.exe` on Windows)
- Python module: `cuda_bind_py.pyd` (Windows) or `cuda_bind_py.so` (Linux/macOS)

## Running the Code

### Standalone CUDA Program

```bash
cd build/Src/cuda_bind/Release
./cuda_bind
```

**Expected Output:**
```
CPU Execution Time = 2280.968018 ms
GPU Execution Time: 35.037186 ms
PASS
count = 1048576
```

### Python with PyBind11

```python
import numpy as np
import cu_matrix_add

# Create 1024×1024 matrices
a = np.random.randint(0, 10, (1024, 1024), dtype=np.int32)
b = np.random.randint(0, 10, (1024, 1024), dtype=np.int32)

# Perform GPU-accelerated matrix multiplication
results = cu_matrix_add.madd(a, b)

# Display results
print(f"CPU Time: {results['cpu_time_ms']:.2f} ms")
print(f"GPU Time: {results['gpu_time_ms']:.2f} ms")
print(f"Speedup: {results['cpu_time_ms'] / results['gpu_time_ms']:.1f}x")
print(f"Validation: {'PASS' if results['pass'] else 'FAIL'}")
```

## Troubleshooting

### Build Issues

**CUDA not found:**
- Ensure CUDA Toolkit is installed and `nvcc` is in your PATH

**Python module import error:**
- Ensure the `.pyd`/`.so` file is in the same directory as your Python script
- Or add the build directory to `sys.path`

**CMake configuration error:**
- Install Python development headers: `python3-dev` (Linux) or ensure Python is installed with dev files (Windows)
- Install PyBind11: `pip install pybind11`

### Performance Issues

If GPU performance is slower than expected:
- Use Release build configuration
- Verify the shared memory kernel is being used
- Check GPU utilization with `nvidia-smi`

## Performance Analysis

### Execution Time Comparison (1024×1024 matrices)

| Implementation | Execution Time | Speedup |
|---------------|----------------|---------|
| CPU (Sequential) | ~2,240 ms | 1x (baseline) |
| GPU (Global Memory) | ~90 ms | ~25x |
| GPU (Shared Memory) | ~35-51 ms | **44-64x** |

### Optimization Techniques

**Shared Memory Benefits:**
- Reduces global memory bandwidth by reusing data within tiles
- Each 16×16 tile is loaded once and reused 16 times
- Significantly faster than global memory access (~100x)

**GPU Configuration:**
- 4,096 blocks (64×64 grid)
- 256 threads per block (16×16)
- Total 1,048,576 threads for parallel computation