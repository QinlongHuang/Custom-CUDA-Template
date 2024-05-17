#include "gemm.h"

__global__
void gemm_gpu_mulblock_no_restrict_kernel(
	int* C,		// [n, m]
	const int* A,	// [n, k]
	const int* B,	// [k, m]
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

void gemm_gpu_mulblock_no_restrict(
	int* C,		// [n, m]
	const int* A,	// [n, k]
	const int* B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
	gemm_gpu_mulblock_no_restrict_kernel<<<m, n>>>(C, A, B, n, m, k);
}