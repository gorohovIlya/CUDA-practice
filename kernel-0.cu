#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

__global__ void kernelAKF(unsigned int currSignal, int* akf, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        double sum = 0.0;
        for (int j = 0; j < n - idx; j++) {
            int a = ((currSignal >> j) & 1) ? 1 : -1;
            int b = ((currSignal >> (j + idx)) & 1) ? 1 : -1;
            sum += a * b;
        }
        akf[idx] = sum;
    }
}

__global__ void generateSignals(unsigned int* signals, int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_combinations = 1u << n;

    if (idx < total_combinations) {
        signals[idx] = idx;
    }
}

int main(int argc, char* argv[])
{
    const int N = 10;
    const unsigned int total_signals = 1 <<  (N - 1);
    int* d_akf0;
    unsigned int* d_signals;
    cudaMalloc(&d_signals, total_signals * sizeof(unsigned int));
    cudaMalloc(&d_akf0, N * sizeof(double));
    int blockSize = 256;
    int gridSize = (total_signals + blockSize - 1) / blockSize;
    generateSignals << <gridSize, blockSize >> > (d_signals, N);
    unsigned int first_signal;
    cudaMemcpy(&first_signal, d_signals, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    kernelAKF << <1, N >> > (first_signal, d_akf0, N);
    unsigned int* signals = new unsigned int[total_signals];
    int* akf0 = new int[N];
    cudaMemcpy(akf0, d_akf0, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(signals, d_signals, total_signals * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(akf0, d_akf0, N * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++) {
        printf("Signal %d: ", i);
        for (int j = N - 1; j >= 0; j--) {

            printf("%d", (signals[i] >> j) & 1);
        }
        printf("\n");
    }
    printf("ACF of 1st signal:\n");
    for (int j = 0; j < N; j++) {
        printf("acf[%d] = %d\n", j, akf0[j]);
    }
    cudaFree(d_signals);
    cudaFree(d_akf0);
    delete[] signals;
    delete[] akf0;
    return 0;
}


//int main(int argc, char* argv[])
//{
//	int deviceCount;
//	cudaDeviceProp devProp;
//	cudaGetDeviceCount(&deviceCount);
//	printf("Found %d devices\n", deviceCount);
//	for (int device = 0; device < deviceCount; device++)
//	{
//		cudaGetDeviceProperties(&devProp, device);
//		printf("Device %d\n", device);
//		printf("Compute capability : %d.%d\n", devProp.major,
//			devProp.minor);
//		printf("Name : %s\n", devProp.name);
//		printf("Total Global Memory : %zu\n",
//			devProp.totalGlobalMem);
//		printf("Total multiprocessors : %d\n",
//			devProp.multiProcessorCount);
//		printf("Shared memory per block: %d\n",
//			devProp.sharedMemPerBlock);
//		printf("Registers per block : %d\n",
//			devProp.regsPerBlock);
//		printf("Warp size : %d\n", devProp.warpSize);
//		printf("Max blocks per multiprocessor : %d\n",
//			devProp.maxBlocksPerMultiProcessor);
//		printf("Max threads per block : %d\n",
//			devProp.maxThreadsPerBlock);
//		printf("Total constant memory : %d\n",
//			devProp.totalConstMem);
//	}
//	return 0;
//}
//#define N (1024 * 1024)
//
//__global__ void kernelAKF(double* signal) {
//
//}
//
//int main(int argc, char* argv[]) {
//    int size_int = sizeof(int);
//    printf("sizeof(int) = %d\n", size_int);
//    return 0;
//}
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
