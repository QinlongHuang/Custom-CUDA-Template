from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="gemm-st", 
    include_dirs=['include'], 
    ext_modules=[
        CUDAExtension(
            "gemmst", 
            ["kernels/gemm_kernel.cu", "pytorch_wrapper/gemm_op_st.cpp"]
        )
    ], 
    cmdclass={
        'build_ext': BuildExtension
    }, 
    version="0.1"
)