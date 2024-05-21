import os
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
gemmjit_module = load(
    name="gemm_jit", 
    extra_include_paths=[os.path.join(module_path, "include")],  # explicitly specify the include path
    sources=[
        os.path.join(module_path, "pytorch_wrapper/gemm_op_jit.cpp"),
        os.path.join(module_path, "kernels/gemm_kernel.cu")
    ],
    verbose=True
)

if __name__ == "__main__":
    import torch
    
    a = torch.randn(4, 4, device="cuda")
    b = torch.randn(4, 4, device="cuda")
    print("PyTorch a@b: \n", a@b)
    
    # JIT (Just-in-time Compilation)
    c = torch.zeros(4, 4, device="cuda")
    gemmjit_module.gemm(c, a, b)
    print("JIT: \n", c)
    
    # Setuptools
    import gemmst
    c = torch.zeros(4, 4, device="cuda")
    gemmst.gemm(c, a, b)
    print("Setuptools: \n", c)
    
    # CMake
    import gemmcm
    c = torch.zeros(4, 4, device="cuda")
    gemmcm.gemm(c, a, b)
    print("CMake: \n", c)