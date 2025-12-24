cmake_minimum_required(VERSION 3.10)
project(cuda_bind_project LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)

file(GLOB HEADERS *.h *.cuh)
source_group("Headers" FILES ${HEADERS})

cuda_add_executable(cuda_bind
    ${HEADERS}
    cuda_bind.cu
    OPTIONS --cudart=shared -lineinfo -arch sm_52 -use_fast_math
    DEBUG -Xptxas -v -g -G
)

target_link_libraries(cuda_bind
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CURAND_LIBRARY}
)