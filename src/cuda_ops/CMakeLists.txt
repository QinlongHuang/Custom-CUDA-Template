cmake_minimum_required(VERSION 3.20)
project(gemm LANGUAGES CXX CUDA)  # activate support for CUDA

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# activate support for PyTorch
# For Conda environment, we need to manually specify the Python include directory and PyTorch's cmake prefix path
find_package(Python COMPONENTS Interpreter Development)  # see https://cmake.org/cmake/help/latest/module/FindPython3.html for more details
message("Python_EXECUTABLE ${Python_EXECUTABLE}")
message("Python_INCLUDE_DIRS ${Python_INCLUDE_DIRS}")
execute_process(COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")
message("PYTORCH_CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH}")
message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
include_directories("${Python_INCLUDE_DIRS}")

# Add pybind11 support
execute_process(COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())" OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

# add custom header files
include_directories("include")

# add source files
file(GLOB SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gemm_kernel.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/pytorch_wrapper/gemm_op.cpp"
)

# generate a shared library rather than an executable application
add_library(gemmcm SHARED ${SOURCES})

# dynamic link library
target_link_libraries(gemmcm PUBLIC ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} pybind11::module)

# set the module name and extension
set_target_properties(gemmcm PROPERTIES 
    PREFIX "${PYTHON_MODULE_PREFIX}" 
    SUFFIX "${PYTHON_MODULE_EXTENSION}" 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)