#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define POLYBENCH_TIME 1

#include "gemm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

// Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

// Declared constant values for ALPHA and BETA
#define ALPHA 32412.0f
#define BETA 2123.0f

// For mixed precision computation
#define FP32_K_PORTION 0.5  // Portion of K loop to compute in FP32

// Tile sizes for shared memory optimization
#define TILE_SIZE 32
#define WARP_SIZE 32

#define RUN_ON_CPU

// Original CPU implementation
void gemm(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta,
          DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
          DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
          DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
    int i, j, k;

    for (i = 0; i < _PB_NI; i++)
    {
        for (j = 0; j < _PB_NJ; j++)
        {
            C[i][j] *= beta;
           
            for (k = 0; k < _PB_NK; ++k)
            {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
}

void init(int ni, int nj, int nk, DATA_TYPE* alpha, DATA_TYPE* beta,
          DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
          DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
          DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
    int i, j;

    *alpha = 32412;
    *beta = 2123;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nk; j++)
        {
            A[i][j] = ((DATA_TYPE) i*j) / NI;
        }
    }

    for (i = 0; i < nk; i++)
    {
        for (j = 0; j < nj; j++)
        {
            B[i][j] = ((DATA_TYPE) i*j) / NI;
        }
    }

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            C[i][j] = ((DATA_TYPE) i*j) / NI;
        }
    }
}

void compareResults(int ni, int nj,
                   DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
                   DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
    int i, j, fail;
    fail = 0;

    // Compare CPU and GPU outputs
    for (i=0; i < ni; i++)
    {
        for (j=0; j < nj; j++)
        {
            if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
                if (fail < 10) {
                    printf("Mismatch at position i=%d, j=%d: CPU=%f, GPU=%f\n",
                          i, j, C[i][j], C_outputFromGpu[i][j]);
                }
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n",
           PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("Setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

// Kernel to compute GEMM using FP32 for the first half of K
__global__ void gemm_fp32_kernel(int ni, int nj, int nk, int k_split,
                               DATA_TYPE alpha, DATA_TYPE beta,
                               DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *partial_c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
   
    if (i < ni && j < nj) {
        float sum = 0.0f;
        for (int k = 0; k < k_split; k++) {
            sum += a[i * nk + k] * b[k * nj + j];
        }
        partial_c[i * nj + j] = sum;
    }
}

// Kernel to compute GEMM using FP64 for the second half of K
__global__ void gemm_fp64_kernel(int ni, int nj, int nk, int k_split,
                               DATA_TYPE alpha, DATA_TYPE beta,
                               DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *partial_c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
   
    if (i < ni && j < nj) {
        double sum = 0.0;
        for (int k = k_split; k < nk; k++) {
            sum += (double)a[i * nk + k] * (double)b[k * nj + j];
        }
        partial_c[i * nj + j] = (float)sum;
    }
}

// Kernel to combine results from FP32 and FP64 computations
__global__ void gemm_combine_kernel(int ni, int nj,
                                  DATA_TYPE alpha, DATA_TYPE beta,
                                  DATA_TYPE *c, DATA_TYPE *partial_fp32, DATA_TYPE *partial_fp64)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
   
    if (i < ni && j < nj) {
        c[i * nj + j] = beta * c[i * nj + j] + alpha * (partial_fp32[i * nj + j] + partial_fp64[i * nj + j]);
    }
}

// Tiled FP32 GEMM kernel with shared memory
__global__ void gemm_fp32_tiled_kernel(int ni, int nj, int nk, int k_split,
                                     DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *partial_c)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
   
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
   
    float sum = 0.0f;
   
    // Loop over all tiles required to compute the block
    for (int t = 0; t < (k_split + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load the matrices from device memory to shared memory
        if (row < ni && t * TILE_SIZE + tx < k_split) {
            As[ty][tx] = a[row * nk + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
       
        if (t * TILE_SIZE + ty < k_split && col < nj) {
            Bs[ty][tx] = b[(t * TILE_SIZE + ty) * nj + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
       
        __syncthreads();
       
        // Compute the partial dot product for this tile
        for (int k = 0; k < TILE_SIZE && t * TILE_SIZE + k < k_split; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
       
        __syncthreads();
    }
   
    // Write the result to the partial_c matrix
    if (row < ni && col < nj) {
        partial_c[row * nj + col] = sum;
    }
}

// Tiled FP64 GEMM kernel with shared memory
__global__ void gemm_fp64_tiled_kernel(int ni, int nj, int nk, int k_split,
                                     DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *partial_c)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
   
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
   
    double sum = 0.0;
   
    // Loop over all tiles required to compute the block
    for (int t = 0; t < (nk - k_split + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load the matrices from device memory to shared memory
        if (row < ni && k_split + t * TILE_SIZE + tx < nk) {
            As[ty][tx] = a[row * nk + k_split + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
       
        if (k_split + t * TILE_SIZE + ty < nk && col < nj) {
            Bs[ty][tx] = b[(k_split + t * TILE_SIZE + ty) * nj + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
       
        __syncthreads();
       
        // Compute the partial dot product for this tile in double precision
        for (int k = 0; k < TILE_SIZE && k_split + t * TILE_SIZE + k < nk; k++) {
            sum += (double)As[ty][k] * (double)Bs[k][tx];
        }
       
        __syncthreads();
    }
   
    // Write the result to the partial_c matrix
    if (row < ni && col < nj) {
        partial_c[row * nj + col] = (float)sum;
    }
}

void gemmCuda(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta,
              DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
              DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
              DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
              DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *B_gpu;
    DATA_TYPE *C_gpu;
    DATA_TYPE *partial_fp32;
    DATA_TYPE *partial_fp64;
   
    // Calculate the split point for mixed precision
    int k_split = (int)(nk * FP32_K_PORTION);
   
    // Allocate device memory
    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
    cudaMalloc((void **)&partial_fp32, sizeof(DATA_TYPE) * NI * NJ);
    cudaMalloc((void **)&partial_fp64, sizeof(DATA_TYPE) * NI * NJ);
   
    // Transfer data from host to device
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
   
    // Define grid and block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((nj + TILE_SIZE - 1) / TILE_SIZE, (ni + TILE_SIZE - 1) / TILE_SIZE);
   
    // Create CUDA streams for parallel execution
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
   
    // Start timer
    polybench_start_instruments;
   
    // Launch kernels in different streams for parallel execution
    gemm_fp32_tiled_kernel<<<grid, block, 0, stream1>>>(ni, nj, nk, k_split, A_gpu, B_gpu, partial_fp32);
    gemm_fp64_tiled_kernel<<<grid, block, 0, stream2>>>(ni, nj, nk, k_split, A_gpu, B_gpu, partial_fp64);
   
    // Wait for both kernels to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
   
    // Combine the results
    gemm_combine_kernel<<<grid, block>>>(ni, nj, alpha, beta, C_gpu, partial_fp32, partial_fp64);
   
    // Check for any kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
   
    cudaDeviceSynchronize();
   
    // Stop and print timer
    printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;
   
    // Transfer results back to host
    cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);
   
    // Cleanup
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
    cudaFree(partial_fp32);
    cudaFree(partial_fp64);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

/* DCE code. Must scan the entire live-out data.
Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
                DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
    int i, j;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++) {
            fprintf(stderr, DATA_PRINTF_MODIFIER, C[i][j]);
            if ((i * ni + j) % 20 == 0) fprintf(stderr, "\n");
        }
    fprintf(stderr, "\n");
}

int main(int argc, char* argv[])
{
    // Retrieve problem size
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    // Variable declaration/allocation
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);

    // Initialize data
    init(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

    GPU_argv_init();

    // Execute GEMM on GPU
    gemmCuda(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

#ifdef RUN_ON_CPU
    // Execute GEMM on CPU for validation
    // Start timer
    polybench_start_instruments;

    gemm(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
   
    // Stop and print timer
    printf("CPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    // Compare CPU and GPU results
    compareResults(ni, nj, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

#else // print output to stderr so no dead code elimination
    print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu));
#endif // RUN_ON_CPU

    // Free allocated memory
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);  
    POLYBENCH_FREE_ARRAY(C);  
    POLYBENCH_FREE_ARRAY(C_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"