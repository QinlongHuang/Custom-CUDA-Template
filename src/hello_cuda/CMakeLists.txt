cmake_minimum_required(VERSION 3.20)
project(gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

include_directories("/usr/local/cuda/targets/x86_64-linux/include")  # for use "cuda_runtime.h"
include_directories("include")

file(GLOB SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

add_executable(gemm ${SOURCES})
target_link_libraries(gemm ${CUDA_LIBRARIES})  # if you call CUDA bulitin function in a non-CUDA file, you need to link this