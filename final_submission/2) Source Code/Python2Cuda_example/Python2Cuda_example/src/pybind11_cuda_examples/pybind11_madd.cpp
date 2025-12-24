// Imports
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <cuda_runtime.h>

// declare external CUDA function (matrix multiplication using GPU shared memory)
extern "C" void launch_matrix_multiplication_shared_memory(int* input1, int* input2, int* output);

// keyword for pybind functionality
namespace py = pybind11;

// python wrapper function for matrix multiplication
py::dict madd_wrapper(py::array_t<int> a1, py::array_t<int> a2)
{
    // matrix size N * N (1024 * 1024)
    const int N = 1024;

    // check matrix dimensions
    if (a1.ndim() != 2 || a2.ndim() != 2)
        throw std::runtime_error("Number of dimensions must be two");

    // check matrix size
    if (a1.shape(0) != N || a1.shape(1) != N ||
        a2.shape(0) != N || a2.shape(1) != N)
        throw std::runtime_error("Input matrices must be 1024x1024");

    // get numpy array buffers
    auto buf1 = a1.request();
    auto buf2 = a2.request();

    // create pointers for matrix data in host (both input matrices)
    int* A = (int*)buf1.ptr;
    int* B = (int*)buf2.ptr;

    // allocate memory for host matrices
    int* reference = new int[N * N];
    int* output_cpu = new int[N * N];

    // CPU matrix multiplication (for reference)
    auto cpuStartTime = std::chrono::high_resolution_clock::now();  // start of execution time
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            reference[row * N + col] = sum;
        }
    }
    auto cpuEndTime = std::chrono::high_resolution_clock::now();    // end of execution time
    std::chrono::duration<float, std::milli> cpuExecutionTime = cpuEndTime - cpuStartTime;  // total execution time

    // declare GPU matrix pointers
    int* d_input1, * d_input2, * d_output;
    int size = N * N * sizeof(int);

    // allocate memory for device matrices
    cudaMalloc(&d_input1, size);
    cudaMalloc(&d_input2, size);
    cudaMalloc(&d_output, size);

    // copy input matrices to device
    cudaMemcpy(d_input1, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, B, size, cudaMemcpyHostToDevice);

    // initialize timing event for measurement
    cudaEvent_t gpuStartTime, gpuEndTime;
    cudaEventCreate(&gpuStartTime);
    cudaEventCreate(&gpuEndTime);

    // launch kernel function on GPU (measure execution time)
    cudaEventRecord(gpuStartTime);
    launch_matrix_multiplication_shared_memory(d_input1, d_input2, d_output);
    cudaEventRecord(gpuEndTime);
    cudaEventSynchronize(gpuEndTime);

    // determine GPU execution time
    float gpuMs = 0;
    cudaEventElapsedTime(&gpuMs, gpuStartTime, gpuEndTime);

    // copy resulting matrix back to host
    auto result = py::array(py::buffer_info(
        nullptr, sizeof(int), py::format_descriptor<int>::value, 2, { N, N },
        { sizeof(int) * N, sizeof(int) }
    ));
    auto buf3 = result.request();
    int* C = (int*)buf3.ptr;
    cudaMemcpy(C, d_output, size, cudaMemcpyDeviceToHost);

    // compare device results to host reference
    bool pass = true;
    int correct_count = 0;
    for (int i = 0; i < N * N; i++) {   // check all N * N elements in array (as 1D matrix)
        if (reference[i] != C[i]) {
            pass = false;
        }
        else {
            correct_count++;
        }
    }

    // clean up memory, events
    delete[] reference;
    delete[] output_cpu;
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
    cudaEventDestroy(gpuStartTime);
    cudaEventDestroy(gpuEndTime);

    // display results
    py::dict results;
    results["cpu_time_ms"] = py::float_(cpuExecutionTime.count());
    results["gpu_time_ms"] = py::float_(gpuMs);
    results["pass"] = py::bool_(pass);
    results["correct_count"] = py::int_(correct_count);

    // return results
    return results;
}

// define python module for pybind11
PYBIND11_MODULE(cu_matrix_add, m) {
    m.doc() = "Pybind11 plugin for CUDA matrix multiplication";
    m.def("madd", &madd_wrapper, "Perform matrix multiplication on GPU, return result and execution time");
}