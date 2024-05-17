#include "gemm.h"
#include <cassert>

constexpr int TILE_SIZE = 32;

__global__
void gemm_gpu_tiling_kernel(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
    __shared__ int a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ int b_tile[TILE_SIZE][TILE_SIZE];
    int res = 0;
    for (int tile_index=0; tile_index < k/TILE_SIZE; tile_index++) {
        // 1. Copy the tile from A and B to shared memory
        a_tile[threadIdx.y][threadIdx.x] = A[(blockIdx.x*TILE_SIZE + threadIdx.y)*k + tile_index*TILE_SIZE + threadIdx.x];
        b_tile[threadIdx.y][threadIdx.x] = B[(tile_index*TILE_SIZE + threadIdx.y)*m + (blockIdx.y*TILE_SIZE + threadIdx.x)];
        __syncthreads();
        // 2. Perform the matrix multiplication
        for (int i=0; i<TILE_SIZE; i++) {
            res += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
        }
        __syncthreads();
    }
	// 3. Copy the result to C
    C[(blockIdx.x*TILE_SIZE + threadIdx.y)*m + (blockIdx.y*TILE_SIZE + threadIdx.x)] = res;
}

void gemm_gpu_tiling(
	int* __restrict__ C,		// [n, m]
	const int* __restrict__ A,	// [n, k]
	const int* __restrict__ B,	// [k, m]
	const int n,
	const int m,
	const int k
) {
    assert (n % TILE_SIZE == 0);
    assert (m % TILE_SIZE == 0);
    assert (k % TILE_SIZE == 0);

    dim3 grid_dim(n/TILE_SIZE, m/TILE_SIZE);
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
	gemm_gpu_tiling_kernel<<<grid_dim, block_dim>>>(C, A, B, n, m, k);
}