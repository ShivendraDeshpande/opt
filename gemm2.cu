#include <cuda.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>

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

#define RUN_ON_CPU

#define FP64_RATIO 0.25  // Fraction of work to be done with FP64

void gemm(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta,
	 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
	 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
	 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
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
	cudaSetDevice( GPU_DEVICE );
}


__global__ void gemm_kernel(int ni, int nj, int nk, float alpha, float beta, float *a, float *b, float *c) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < ni) && (j < nj)) {
        float sum_fp32 = 0.0f;
        double sum_fp64 = 0.0; // Added FP64 accumulator

        // Hybrid FP32/FP64 computation
        for (int k = 0; k < nk; k++) {
            if (k < (int)(nk * FP64_RATIO)) {  // Use FP64 for the first FP64_RATIO fraction of the loop
                sum_fp64 += (double)a[i * nk + k] * (double)b[k * nj + j];  // Cast to double for FP64 calculation
            } else {
                sum_fp32 += a[i * nk + k] * b[k * nj + j]; // Use FP32 for the rest
            }
        }

        // Combine the results (convert FP64 back to FP32)
        c[i * nj + j] = beta * c[i * nj + j] + alpha * (sum_fp32 + (float)sum_fp64);
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

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);

	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil( ((float)NI)/ ((float)block.x) )),(size_t)(ceil( ((float)NJ)/ ((float)block.y) )));

	/* Start timer. */
  	polybench_start_instruments;

	gemm_kernel<<< grid, block >>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
	cudaThreadSynchronize();

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