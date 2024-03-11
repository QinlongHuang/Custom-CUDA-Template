#include <torch/extension.h>
#include "include/add_op.h"  // claim the op with .h file


void torch_launch_add(torch::Tensor &c, 
    const torch::Tensor &a, 
    const torch::Tensor &b, 
    int n) {
    add_op((float *)c.data_ptr(), (const float *)a.data_ptr(), (const float *)b.data_ptr(), n);
}


// define python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add", &torch_launch_add, "launch add wrapper");
}