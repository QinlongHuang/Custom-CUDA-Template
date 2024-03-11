# Code Template for Custom CUDA operations
Several simple examples for PyTorch calling custom CUDA operators.

Compile the CUDA kernels and their cpp wrappers using JIT(Just-In-Time).

For more accurate time statistics, you'd best use `nvprof` or `nsys` to run the code.

## Environments
- OS: Ubuntu 20.04
- GPU: NVIDIA RTX 4090 w/ Driver 525.147.05
- CUDA: 11.8
- Python: 3.8.18
- PyTorch: 2.1.2+cu118
- CMake: 3.26.0-rc6
- Ninja: 1.10.0
- GCC: 9.4.0

Cannot ensure successful running in other environments.

## Code structure
```
📦Custom-CUDA-Template
 ┣ 📂src
 ┃ ┗ 📂cuda_ops
 ┃ ┃ ┣ 📂include
 ┃ ┃ ┃ ┗ 📜add_op.h
 ┃ ┃ ┣ 📂kernels
 ┃ ┃ ┃ ┣ 📜add_kernel.cu
 ┃ ┃ ┃ ┗ 📜fused_leaky_relu_kernel.cu
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜add.py
 ┃ ┃ ┣ 📜add_op.cpp
 ┃ ┃ ┣ 📜fused_leaky_relu.py
 ┃ ┃ ┗ 📜fused_leaky_relu_op.cpp
 ┣ 📜.project-root
 ┗ 📜README.md
```

## Reference
- [godweiyang/NN-CUDA-Example](https://github.com/godweiyang/NN-CUDA-Example)
- [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)