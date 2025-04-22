#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "gemm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

// Define block sizes for better performance
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Define the number of warps per block
#define WARPS_PER_BLOCK ((BLOCK_SIZE_X * BLOCK_SIZE_Y) / 32)

// Define the number of warps to assign to FP64 computation
#define FP64_WARPS (WARPS_PER_BLOCK / 2)

#define RUN_ON_CPU

void gemm(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
    int i,j,k;

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

void init(int ni, int nj, int nk, DATA_TYPE* alpha, DATA_TYPE* beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
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

void compareResults(int ni, int nj, DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
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
            }
        }
    }

    // Print results
    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

// Optimized GEMM kernel with FP32/FP64 overlap using warp specialization
__global__ void gemm_kernel_fp32_fp64_overlap(int ni, int nj, int nk, 
                                             DATA_TYPE alpha, DATA_TYPE beta,
                                             DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Global row and column indices
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    // Calculate warp ID within the block
    int warpId = (ty * blockDim.x + tx) / 32;
    
    // Determine if this warp should use FP64 computation
    bool useFP64 = (warpId < FP64_WARPS);
    
    // Shared memory for intermediate results
    __shared__ float results_fp32[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ double results_fp64[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    
    // Check if this thread is within matrix bounds
    if (row < ni && col < nj) {
        if (useFP64) {
            // FP64 computation path
            double c_val = (double)c[row * nj + col];
            c_val *= (double)beta;
            
            // Compute matrix multiplication using FP64
            for (int k = 0; k < nk; k++) {
                double a_val = (double)a[row * nk + k];
                double b_val = (double)b[k * nj + col];
                c_val += (double)alpha * a_val * b_val;
            }
            
            // Store result in shared memory
            results_fp64[ty][tx] = c_val;
        } else {
            // FP32 computation path
            float c_val = (float)c[row * nj + col];
            c_val *= (float)beta;
            
            // Compute matrix multiplication using FP32
            for (int k = 0; k < nk; k++) {
                float a_val = (float)a[row * nk + k];
                float b_val = (float)b[k * nj + col];
                c_val += (float)alpha * a_val * b_val;
            }
            
            // Store result in shared memory
            results_fp32[ty][tx] = c_val;
        }
    }
    
    // Synchronize to ensure all warps have completed their computations
    __syncthreads();
    
    // Write results back to global memory
    if (row < ni && col < nj) {
        if (useFP64) {
            c[row * nj + col] = (DATA_TYPE)results_fp64[ty][tx];
        } else {
            c[row * nj + col] = (DATA_TYPE)results_fp32[ty][tx];
        }
    }
}

void gemmCuda(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *B_gpu;
    DATA_TYPE *C_gpu;

    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);

    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid((nj + block.x - 1) / block.x, (ni + block.y - 1) / block.y);

    /* Start timer. */
    polybench_start_instruments;

    // Launch the optimized kernel with FP32/FP64 overlap
    gemm_kernel_fp32_fp64_overlap<<<grid, block>>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
    cudaDeviceSynchronize();

    /* Stop and print timer. */
    printf("GPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);    

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
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

int main(int argc, char *argv[])
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
    POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
    POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
    POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu,DATA_TYPE,NI,NJ,ni,nj);

    init(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

    GPU_argv_init();

    gemmCuda(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

#ifdef RUN_ON_CPU
    /* Start timer. */
    polybench_start_instruments;

    gemm(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

    /* Stop and print timer. */
    printf("CPU Time in seconds:\n");
    polybench_stop_instruments;
    polybench_print_instruments;

    compareResults(ni, nj, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

#else //print output to stderr so no dead code elimination

    print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu));

#endif //RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);  
    POLYBENCH_FREE_ARRAY(C);  
    POLYBENCH_FREE_ARRAY(C_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
