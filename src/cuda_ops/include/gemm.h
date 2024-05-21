#pragma once

void gemm_cpu_simd(
	float* __restrict__ C,		// [n, m]
	const float* __restrict__ A,	// [n, k]
	const float* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
);

void gemm_gpu_mulblock(
	float* __restrict__ C,		// [n, m]
	const float* __restrict__ A,	// [n, k]
	const float* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
);