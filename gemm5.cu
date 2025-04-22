/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

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
 
 // Block size configuration for better performance
 #define BLOCK_SIZE_X 16
 #define BLOCK_SIZE_Y 16
 
 // Number of warps per block
 #define WARPS_PER_BLOCK ((BLOCK_SIZE_X * BLOCK_SIZE_Y) / 32)
 
 // Number of warps to assign to FP64 computation
 #define FP64_WARPS (WARPS_PER_BLOCK / 2)
 
 // Tile size for shared memory optimization
 #define TILE_SIZE 16
 
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
     printf("Setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
     cudaSetDevice(GPU_DEVICE);
 }
 
 // FP32 GEMM kernel for stream 1
 __global__ void gemm_kernel_fp32(int ni, int nj, int nk, float alpha, float beta, 
                                float *a, float *b, float *c)
 {
     // Shared memory for tiling
     __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
     __shared__ float b_tile[TILE_SIZE][TILE_SIZE];
     
     int bx = blockIdx.x;
     int by = blockIdx.y;
     int tx = threadIdx.x;
     int ty = threadIdx.y;
     
     // Global row and column indices
     int row = by * TILE_SIZE + ty;
     int col = bx * TILE_SIZE + tx;
     
     // Accumulator for dot product
     float sum = 0.0f;
     
     // Loop over tiles
     for (int t = 0; t < (nk + TILE_SIZE - 1) / TILE_SIZE; t++) {
         // Load tiles into shared memory
         if (row < ni && t * TILE_SIZE + tx < nk) {
             a_tile[ty][tx] = a[row * nk + t * TILE_SIZE + tx];
         } else {
             a_tile[ty][tx] = 0.0f;
         }
         
         if (t * TILE_SIZE + ty < nk && col < nj) {
             b_tile[ty][tx] = b[(t * TILE_SIZE + ty) * nj + col];
         } else {
             b_tile[ty][tx] = 0.0f;
         }
         
         __syncthreads();
         
         // Compute dot product for this tile
         for (int k = 0; k < TILE_SIZE; k++) {
             sum += a_tile[ty][k] * b_tile[k][tx];
         }
         
         __syncthreads();
     }
     
     // Write result to global memory
     if (row < ni && col < nj) {
         c[row * nj + col] = alpha * sum + beta * c[row * nj + col];
     }
 }
 
 // FP64 GEMM kernel for stream 2
 __global__ void gemm_kernel_fp64(int ni, int nj, int nk, double alpha, double beta, 
                                double *a, double *b, double *c)
 {
     // Shared memory for tiling
     __shared__ double a_tile[TILE_SIZE][TILE_SIZE];
     __shared__ double b_tile[TILE_SIZE][TILE_SIZE];
     
     int bx = blockIdx.x;
     int by = blockIdx.y;
     int tx = threadIdx.x;
     int ty = threadIdx.y;
     
     // Global row and column indices
     int row = by * TILE_SIZE + ty;
     int col = bx * TILE_SIZE + tx;
     
     // Accumulator for dot product
     double sum = 0.0;
     
     // Loop over tiles
     for (int t = 0; t < (nk + TILE_SIZE - 1) / TILE_SIZE; t++) {
         // Load tiles into shared memory
         if (row < ni && t * TILE_SIZE + tx < nk) {
             a_tile[ty][tx] = a[row * nk + t * TILE_SIZE + tx];
         } else {
             a_tile[ty][tx] = 0.0;
         }
         
         if (t * TILE_SIZE + ty < nk && col < nj) {
             b_tile[ty][tx] = b[(t * TILE_SIZE + ty) * nj + col];
         } else {
             b_tile[ty][tx] = 0.0;
         }
         
         __syncthreads();
         
         // Compute dot product for this tile
         for (int k = 0; k < TILE_SIZE; k++) {
             sum += a_tile[ty][k] * b_tile[k][tx];
         }
         
         __syncthreads();
     }
     
     // Write result to global memory
     if (row < ni && col < nj) {
         c[row * nj + col] = alpha * sum + beta * c[row * nj + col];
     }
 }
 
 // Warp-specialized GEMM kernel that utilizes both FP32 and FP64 compute units
 __global__ void gemm_kernel_warp_specialized(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, 
                                           DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
 {
     // Shared memory for tiles
     __shared__ float a_tile_fp32[TILE_SIZE][TILE_SIZE];
     __shared__ float b_tile_fp32[TILE_SIZE][TILE_SIZE];
     __shared__ double a_tile_fp64[TILE_SIZE][TILE_SIZE];
     __shared__ double b_tile_fp64[TILE_SIZE][TILE_SIZE];
     
     int bx = blockIdx.x;
     int by = blockIdx.y;
     int tx = threadIdx.x;
     int ty = threadIdx.y;
     
     // Global row and column indices
     int row = by * TILE_SIZE + ty;
     int col = bx * TILE_SIZE + tx;
     
     // Calculate warp ID
     int warp_id = (ty * blockDim.x + tx) / 32;
     
     // Determine if this warp should use FP64 computation
     bool use_fp64 = (warp_id < FP64_WARPS);
     
     // Accumulators for dot product
     float sum_fp32 = 0.0f;
     double sum_fp64 = 0.0;
     
     // Loop over tiles
     for (int t = 0; t < (nk + TILE_SIZE - 1) / TILE_SIZE; t++) {
         // Load tiles into shared memory
         if (row < ni && t * TILE_SIZE + tx < nk) {
             if (use_fp64) {
                 a_tile_fp64[ty][tx] = (double)a[row * nk + t * TILE_SIZE + tx];
             } else {
                 a_tile_fp32[ty][tx] = (float)a[row * nk + t * TILE_SIZE + tx];
             }
         } else {
             if (use_fp64) {
                 a_tile_fp64[ty][tx] = 0.0;
             } else {
                 a_tile_fp32[ty][tx] = 0.0f;
             }
         }
         
         if (t * TILE_SIZE + ty < nk && col < nj) {
             if (use_fp64) {
                 b_tile_fp64[ty][tx] = (double)b[(t * TILE_SIZE + ty) * nj + col];
             } else {
                 b_tile_fp32[ty][tx] = (float)b[(t * TILE_SIZE + ty) * nj + col];
             }
         } else {
             if (use_fp64) {
                 b_tile_fp64[ty][tx] = 0.0;
             } else {
                 b_tile_fp32[ty][tx] = 0.0f;
             }
         }
         
         __syncthreads();
         
         // Compute dot product for this tile
         if (use_fp64) {
             for (int k = 0; k < TILE_SIZE; k++) {
                 sum_fp64 += a_tile_fp64[ty][k] * b_tile_fp64[k][tx];
             }
         } else {
             for (int k = 0; k < TILE_SIZE; k++) {
                 sum_fp32 += a_tile_fp32[ty][k] * b_tile_fp32[k][tx];
             }
         }
         
         __syncthreads();
     }
     
     // Write result to global memory
     if (row < ni && col < nj) {
         if (use_fp64) {
             c[row * nj + col] = (DATA_TYPE)(alpha * sum_fp64 + beta * (double)c[row * nj + col]);
         } else {
             c[row * nj + col] = (DATA_TYPE)(alpha * sum_fp32 + beta * (float)c[row * nj + col]);
         }
     }
 }
 
 void gemmCuda(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), 
     DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
 {
     // Create CUDA streams for concurrent execution
     cudaStream_t stream1, stream2;
     cudaStreamCreate(&stream1);
     cudaStreamCreate(&stream2);
     
     // Allocate device memory
     DATA_TYPE *A_gpu, *B_gpu, *C_gpu;
     float *A_fp32, *B_fp32, *C_fp32;
     double *A_fp64, *B_fp64, *C_fp64;
     
     // Allocate memory for original data
     cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
     cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
     cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
     
     // Allocate memory for FP32 data
     cudaMalloc((void **)&A_fp32, sizeof(float) * NI * NK);
     cudaMalloc((void **)&B_fp32, sizeof(float) * NK * NJ);
     cudaMalloc((void **)&C_fp32, sizeof(float) * NI * NJ);
     
     // Allocate memory for FP64 data
     cudaMalloc((void **)&A_fp64, sizeof(double) * NI * NK);
     cudaMalloc((void **)&B_fp64, sizeof(double) * NK * NJ);
     cudaMalloc((void **)&C_fp64, sizeof(double) * NI * NJ);
     
     // Copy data to device
     cudaMemcpyAsync(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice, stream1);
     cudaMemcpyAsync(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice, stream1);
     cudaMemcpyAsync(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice, stream1);
     
     // Convert data to FP32 and FP64 (can be done in parallel in separate streams)
     dim3 convert_block(256);
     dim3 convert_grid((NI * NK + convert_block.x - 1) / convert_block.x);
     
     // Define kernel configuration
     dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
     dim3 grid((nj + block.x - 1) / block.x, (ni + block.y - 1) / block.y);
     
     /* Start timer. */
     polybench_start_instruments;
     
     // Method 1: Use warp specialization within a single kernel
     gemm_kernel_warp_specialized<<<grid, block>>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
     
     // Method 2: Use separate streams for FP32 and FP64 computation
     // Convert data types
     for (int i = 0; i < ni; i++) {
         for (int j = 0; j < nk; j++) {
             A_fp32[i * nk + j] = (float)A[i][j];
             A_fp64[i * nk + j] = (double)A[i][j];
         }
     }
     
     for (int i = 0; i < nk; i++) {
         for (int j = 0; j < nj; j++) {
             B_fp32[i * nj + j] = (float)B[i][j];
             B_fp64[i * nj + j] = (double)B[i][j];
         }
     }
     
     for (int i = 0; i < ni; i++) {
         for (int j = 0; j < nj; j++) {
             C_fp32[i * nj + j] = (float)C[i][j];
             C_fp64[i * nj + j] = (double)C[i][j];
         }
     }
     
     // Launch FP32 kernel in stream1
     gemm_kernel_fp32<<<grid, block, 0, stream1>>>(ni, nj, nk, (float)alpha, (float)beta, 
                                                A_fp32, B_fp32, C_fp32);
     
     // Launch FP64 kernel in stream2
     gemm_kernel_fp64<<<grid, block, 0, stream2>>>(ni, nj, nk, (double)alpha, (double)beta, 
                                                A_fp64, B_fp64, C_fp64);
     
     // Synchronize streams
     cudaStreamSynchronize(stream1);
     cudaStreamSynchronize(stream2);
     
     // Combine results from FP32 and FP64 kernels (average them)
     for (int i = 0; i < ni; i++) {
         for (int j = 0; j < nj; j++) {
             C_gpu[i * nj + j] = (DATA_TYPE)((C_fp32[i * nj + j] + C_fp64[i * nj + j]) / 2.0);
         }
     }
     
     cudaDeviceSynchronize();
     
     /* Stop and print timer. */
     printf("GPU Time in seconds:\n");
     polybench_stop_instruments;
     polybench_print_instruments;
     
     // Copy result back to host
     cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);
     
     // Free device memory
     cudaFree(A_gpu);
     cudaFree(B_gpu);
     cudaFree(C_gpu);
     cudaFree(A_fp32);
     cudaFree(B_fp32);
     cudaFree(C_fp32);
     cudaFree(A_fp64);
     cudaFree(B_fp64);
     cudaFree(C_fp64);
     
     // Destroy streams
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
     fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
     if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
     }
   fprintf (stderr, "\n");
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
 