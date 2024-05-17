#pragma once

void gemm_cpu_naive(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
);

void gemm_cpu_simd(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
);

void gemm_gpu_1thread(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
);

void gemm_gpu_multhread(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
);

void gemm_gpu_mulblock(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
);

void gemm_gpu_mulblock_no_restrict(
	int* C,		// [n, m]
	const int* A,	// [n, k]
	const int* B,	// [k, m]
	const int n,
	const int m,
	const int k
);

void gemm_gpu_mulblock_no_restrict_reg(
	int* C,		// [n, m]
	const int* A,	// [n, k]
	const int* B,	// [k, m]
	const int n,
	const int m,
	const int k
);

void gemm_gpu_mulblock_reg(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
);

void gemm_gpu_tiling(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
);
