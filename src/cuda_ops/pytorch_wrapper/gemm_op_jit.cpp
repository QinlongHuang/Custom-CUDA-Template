#include <torch/extension.h>
#include "gemm.h"  // claim the op with .h file

void gemm_torch_warpper(
    torch::Tensor &c,  // [n, m]
    const torch::Tensor &a,  // [n, k]
    const torch::Tensor &b) { // [k, m]
    int n = c.size(0);
    int m = c.size(1);
    int k = a.size(1);
    gemm_gpu_mulblock((float *)c.data_ptr(), (const float *)a.data_ptr(), (const float *)b.data_ptr(), n, m, k);
}

// define python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_torch_warpper, "launch gemm_gpu_mulblock");
}

