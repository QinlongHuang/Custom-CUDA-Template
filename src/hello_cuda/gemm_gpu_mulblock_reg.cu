#include "gemm.h"

__global__
void gemm_gpu_mulblock_reg_kernel(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
	const int i = threadIdx.x;
    const int j = blockIdx.x;
    int res = 0;
    for (int l = 0; l < k; l++) {
        res += A[i*k + l] * B[l*m + j];
    }
    C[i*m + j] = res;
}

void gemm_gpu_mulblock_reg(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
	gemm_gpu_mulblock_reg_kernel<<<m, n>>>(C, A, B, n, m, k);
}