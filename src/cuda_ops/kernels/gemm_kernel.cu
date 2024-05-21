#include "gemm.h"

__global__
void gemm_gpu_mulblock_kernel(
	float* __restrict__ C,		// [n, m]
	const float* __restrict__ A,	// [n, k]
	const float* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
	const int i = threadIdx.x;
    const int j = blockIdx.x;
    for (int l = 0; l < k; l++) {
        C[i*m + j] += A[i*k + l] * B[l*m + j];
    }
}

void gemm_gpu_mulblock(
	float* __restrict__ C,		// [n, m]
	const float* __restrict__ A,	// [n, k]
	const float* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
	gemm_gpu_mulblock_kernel<<<m, n>>>(C, A, B, n, m, k);
}