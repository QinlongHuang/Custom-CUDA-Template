#include "gemm.h"

__attribute__((optimize("O1")))	// Enforce O1 optimization to avoid loop unrolling and SIMD
void gemm_cpu_naive(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
	for (int l=0; l<k; l++)
		for (int i=0; i<n; i++)
			for (int j=0; j<m; j++)
				C[i * m + j] += A[i * k + l] * B[l * m + j];
}

__attribute__((optimize("O3")))	// Enforce O3 optimization to use loop unrolling and SIMD
void gemm_cpu_simd(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
	for (int l=0; l<k; l++)
		for (int i=0; i<n; i++)
			for (int j=0; j<m; j++)
				C[i * m + j] += A[i * k + l] * B[l * m + j];
}