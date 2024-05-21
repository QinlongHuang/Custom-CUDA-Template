# Code Template for Custom CUDA operations
Several simple examples for CUDA C++ (`hello_cuda`) and PyTorch calling custom CUDA operators (`cuda_ops`).

For CUDA C++ programs, use CMake or Ninja to build.
```bash
# Modern CMake(>= 3.8)
# CMake
cmake -S . -B build
cmake --build build

# Ninja
cmake -S . -Bbuild -G Ninja
cmake --build build
```

For PyTorch custom CUDA operators, see `src/cuda_ops`.

You can run `nvprof` or `nsys` to profile your CUDA ops, e.g., 
```bash
nsys profile src/cuda_ops/add.py
```
And then open the generated `*.nsys-rep` file in [Nsight Systems](https://developer.nvidia.com/nsight-systems).

TODO: For more tutorials, see `docs`.

## Environments
- OS: Ubuntu 20.04
- GPU: NVIDIA RTX 4090 w/ Driver 525.147.05
- CUDA: 11.8
- Python: 3.8.18
- PyTorch: 2.1.2+cu118
- CMake: 3.26.0-rc6
- Ninja: 1.10.0
- GCC: 9.4.0

**No guarantee to other environments.**

## Code structure
```
📦Custom-CUDA-Template
 ┣ 📂docs
 ┃ ┣ 📜CUDA编程模型简介.pdf
 ┃ ┗ 📜自定义CUDA算子用于PyTorch.pdf
 ┣ 📂src
 ┃ ┣ 📂cuda_ops
 ┃ ┃ ┣ 📂include
 ┃ ┃ ┃ ┗ 📜gemm.h
 ┃ ┃ ┣ 📂kernels
 ┃ ┃ ┃ ┣ 📜fused_leaky_relu_kernel.cu
 ┃ ┃ ┃ ┗ 📜gemm_kernel.cu
 ┃ ┃ ┣ 📂pytorch_wrapper
 ┃ ┃ ┃ ┣ 📜fused_leaky_relu_op.cpp
 ┃ ┃ ┃ ┣ 📜gemm_op.cpp
 ┃ ┃ ┃ ┣ 📜gemm_op_jit.cpp
 ┃ ┃ ┃ ┗ 📜gemm_op_st.cpp
 ┃ ┃ ┣ 📜CMakeLists.txt
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜add.py
 ┃ ┃ ┣ 📜fused_leaky_relu.py
 ┃ ┃ ┣ 📜readme.md
 ┃ ┃ ┣ 📜setup.py
 ┃ ┃ ┗ 📜test.py
 ┃ ┗ 📂hello_cuda
 ┃ ┃ ┣ 📂include
 ┃ ┃ ┃ ┗ 📜gemm.h
 ┃ ┃ ┣ 📜CMakeLists.txt
 ┃ ┃ ┣ 📜gemm_cpu.cpp
 ┃ ┃ ┣ 📜gemm_gpu_1thread.cu
 ┃ ┃ ┣ 📜gemm_gpu_mulblock.cu
 ┃ ┃ ┣ 📜gemm_gpu_mulblock_no_restrict.cu
 ┃ ┃ ┣ 📜gemm_gpu_mulblock_no_restrict_reg.cu
 ┃ ┃ ┣ 📜gemm_gpu_mulblock_reg.cu
 ┃ ┃ ┣ 📜gemm_gpu_multhread.cu
 ┃ ┃ ┣ 📜gemm_gpu_tiling.cu
 ┃ ┃ ┗ 📜main.cpp
 ┣ 📜.gitignore
 ┣ 📜.project-root
 ┣ 📜LICENSE
 ┗ 📜README.md
```

## Reference
- [godweiyang/NN-CUDA-Example](https://github.com/godweiyang/NN-CUDA-Example)
- [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)
- [interestingLSY/CUDA-From-Correctness-To-Performance-Code](https://github.com/interestingLSY/CUDA-From-Correctness-To-Performance-Code/)

## Recommendation
- [NVIDIA CUDA 编程指南](https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf)
- [HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese)
- [CUDA C++ 编程指北 官方翻译校验版本 - 编程相关 - 老潘的AI社区](https://ai.oldpan.me/t/topic/72)