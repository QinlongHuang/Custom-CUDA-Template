#include "gemm.h"

__global__
void gemm_gpu_1thread_kernel(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
	for (int i=0; i<n; i++) 
		for (int j=0; j<m; j++) {
			int sum = 0;
			for (int l=0; l<k; l++) 
				sum += A[i*k + l] * B[l*m + j];
			C[i*m + j] = sum;
		}
}

void gemm_gpu_1thread(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
	gemm_gpu_1thread_kernel<<<1, 1>>>(C, A, B, n, m, k);
}