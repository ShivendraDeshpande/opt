/**
 * gemm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gold <sgrauerg@gmail.com>
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
 #include <cuda_runtime.h> // Include for CUDA functions
 #include <device_launch_parameters.h> // Include for __global__ etc.
 #include <math.h> // Include for ceil
 
 #define POLYBENCH_TIME 1
 
 #include "gemm.cuh" // Contains DATA_TYPE, NI, NJ, NK, etc.
 #include "../../common/polybench.h"
 #include "../../common/polybenchUtilFuncts.h"
 
 #define GPU_DEVICE 0
 
 //define the error threshold for the results "not matching"
 #define PERCENT_DIFF_ERROR_THRESHOLD 0.05
 
 /* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
 // Use double for alpha and beta for accuracy in mixed precision
 #define ALPHA 32412.0
 #define BETA 2123.0
 
 #define RUN_ON_CPU
 
 // Define tile sizes for the tiled GEMM kernel
 // These should be tuned for the specific GPU architecture
 // TILE_M and TILE_N determine the block dimensions (TILE_N in x, TILE_M in y)
 // TILE_K determines the size of the shared memory tiles in the K dimension
 #define TILE_M 32
 #define TILE_N 32
 #define TILE_K 16 // Often smaller due to shared memory capacity
 
 // Original CPU gemm function (for verification)
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
 
     // Use the defined double constants
     *alpha = ALPHA;
     *beta = BETA;
 
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
             // Use percentDiff for floating point comparison
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
     cudaSetDevice( GPU_DEVICE );
 }
 
 
 // Tiled GEMM Kernel with Shared Memory and Mixed Precision (FP32 Multiply, FP64 Accumulate)
 __global__ void gemm_kernel_tiled_mixed_precision(int ni, int nj, int nk, double alpha, double beta, DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
 {
     // Thread block dimensions (derived from TILE_N and TILE_M)
     // threadIdx.x goes from 0 to TILE_N - 1
     // threadIdx.y goes from 0 to TILE_M - 1
 
     // Block indices
     const int block_m = blockIdx.y; // Block row index in C
     const int block_n = blockIdx.x; // Block column index in C
 
     // Thread indices within the block
     const int thread_m = threadIdx.y; // Thread row index within block (0 to TILE_M-1)
     const int thread_n = threadIdx.x; // Thread column index within block (0 to TILE_N-1)
 
     // Global indices for the element of C this thread is responsible for
     const int global_m = block_m * TILE_M + thread_m;
     const int global_n = block_n * TILE_N + thread_n;
 
     // Accumulator for C[global_m][global_n], using double for accuracy
     double c_val_accumulator = 0.0;
 
     // Declare shared memory for tiles of A and B
     // Use float for shared memory to match FP32 multiplication
     // Add padding (+1) to the K dimension to potentially avoid bank conflicts
     __shared__ float sA[TILE_M][TILE_K + 1];
     __shared__ float sB[TILE_K][TILE_N + 1];
 
     // Loop over the K dimension (tiles of K)
     // Each iteration processes a TILE_K x TILE_K block product
     for (int k_tile_start = 0; k_tile_start < nk; k_tile_start += TILE_K)
     {
         // --- Cooperative Loading into Shared Memory ---
         // Threads load a tile of A and a tile of B into shared memory
         // Each thread loads one element per iteration of the outer K loop
 
         // Calculate global indices for loading A tile
         // sA[thread_m][thread_n] corresponds to A[global_a_m][global_a_k]
         const int global_a_m = block_m * TILE_M + thread_m;
         const int global_a_k = k_tile_start + thread_n; // thread_n iterates along K for A
 
         // Calculate global indices for loading B tile
         // sB[thread_m][thread_n] corresponds to B[global_b_k][global_b_n]
         const int global_b_k = k_tile_start + thread_m; // thread_m iterates along K for B
         const int global_b_n = block_n * TILE_N + thread_n;
 
         // Check bounds before loading from global memory
         // Load from global memory (DATA_TYPE, assumed double), cast to float for shared memory
         if (global_a_m < ni && global_a_k < nk) {
             sA[thread_m][thread_n] = (float)a[global_a_m * NK + global_a_k];
         } else {
             sA[thread_m][thread_n] = 0.0f; // Pad with zero if out of bounds
         }
 
         if (global_b_k < nk && global_b_n < nj) {
             sB[thread_m][thread_n] = (float)b[global_b_k * NJ + global_b_n];
         } else {
             sB[thread_m][thread_n] = 0.0f; // Pad with zero if out of bounds
         }
 
         // Synchronize to ensure all threads in the block have loaded their shared memory tiles
         // before any thread starts using the shared data for computation.
         __syncthreads();
 
         // --- Tile Multiplication and Accumulation ---
         // Perform matrix multiplication of the loaded tiles in shared memory
         // This is where FP32 and FP64 units are used and can overlap via pipelining/interleaving
 
         for (int k_inner = 0; k_inner < TILE_K; ++k_inner)
         {
             // Get elements from shared memory (FP32)
             float a_elem_fp32 = sA[thread_m][k_inner];
             float b_elem_fp32 = sB[k_inner][thread_n];
 
             // Perform multiplication in FP32
             // This instruction targets FP32 Multiply Units
             float product_fp32 = a_elem_fp32 * b_elem_fp32;
 
             // Accumulate into the FP64 accumulator
             // Cast FP32 product to FP64
             // This instruction (addition) targets FP64 Add Units
             c_val_accumulator += (double)product_fp32;
         }
 
         // Synchronize to ensure all threads have finished using the current shared memory tiles
         // before the next tiles are loaded in the outer loop.
         __syncthreads();
     }
 
     // --- Final Scaling and Write Back ---
     // Apply alpha and beta scaling using double precision
     // This uses FP64 Multiply/Add Units and Global Memory Store Units
     if (global_m < ni && global_n < nj)
     {
         // Load initial C value from global memory (DATA_TYPE, assumed double)
         double initial_c_double = c[global_m * NJ + global_n];
 
         // Apply beta scaling using double
         initial_c_double *= beta; // Uses FP64 Multiply Units
 
         // Add the accumulated A*B product (scaled by alpha)
         // alpha * c_val_accumulator uses FP64 Multiply Units
         // The final addition uses FP64 Add Units
         c[global_m * NJ + global_n] = initial_c_double + (alpha * c_val_accumulator);
 
         // Store result back to global memory (DATA_TYPE, assumed double)
         // Uses Global Memory Store Units
     }
 }
 
 
 void gemmCuda(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
     DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(C_outputFromGpu,NI,NJ,ni,nj))
 {
     DATA_TYPE *A_gpu;
     DATA_TYPE *B_gpu;
     DATA_TYPE *C_gpu;
 
     // Allocate memory on the GPU
     cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
     cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
     cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);
 
     // Copy data from host to device
     cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
     cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
     cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);
 
     // Define block and grid dimensions for the tiled kernel
     // Block dimensions are TILE_N in x and TILE_M in y
     dim3 block(TILE_N, TILE_M);
     // Grid dimensions cover the entire C matrix (NI x NJ) using TILE_M x TILE_N blocks
     dim3 grid((size_t)(ceil( ((float)NJ)/ ((float)TILE_N) )),(size_t)(ceil( ((float)NI)/ ((float)TILE_M) )));
 
     /* Start timer. */
       polybench_start_instruments;
 
     // Launch the tiled, mixed-precision kernel
     // Pass alpha and beta as double
     gemm_kernel_tiled_mixed_precision<<< grid, block >>>(ni, nj, nk, (double)alpha, (double)beta, A_gpu, B_gpu, C_gpu);
     cudaDeviceSynchronize(); // Use cudaDeviceSynchronize for more robust timing
 
     /* Stop and print timer. */
     printf("GPU Time in seconds:\n");
       polybench_stop_instruments;
      polybench_print_instruments;
 
     // Copy results back from device to host
     cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);
 
     // Free GPU memory
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
 
     // Call the CUDA version with the tiled kernel
     gemmCuda(ni, nj, nk, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));
 
 
     #ifdef RUN_ON_CPU
 
         /* Start timer. */
           polybench_start_instruments;
 
         // Re-initialize C for CPU computation as gemmCuda modified it
         init(ni, nj, nk, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C)); // Re-init C
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