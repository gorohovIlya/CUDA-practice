
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>


__global__ void generateAndFindAKF(size_t* goodSignals, int* sidelobes, int n) {
    size_t signal = blockIdx.x;
    extern __shared__ int s_akf[];

    if (threadIdx.x < n) {
        s_akf[threadIdx.x] = 0;
    }
    __syncthreads();

    int threadBit = ((signal >> threadIdx.x) & 1) ? 1 : -1;
    for (int shift = 0; shift < n; shift++) {
        if (threadIdx.x >= shift) {
            int paired_bit = ((signal >> (threadIdx.x - shift)) & 1) ? 1 : -1;
            atomicAdd(&s_akf[shift], threadBit * paired_bit);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int max_sidelobe = 0;
        for (int i = 1; i < n; i++) {
            int sidelobe = s_akf[i];
            if (sidelobe < 0) {
                sidelobe *= -1;
            }
            if (sidelobe > max_sidelobe) {
                max_sidelobe = sidelobe;
            }
        }

        if (n > 24) {
            if (max_sidelobe < 3 && (((signal & 0b111) == 0b111) || ((signal >> (n - 3) == 0b111)))) {
                goodSignals[blockIdx.x] = signal;
                sidelobes[blockIdx.x] = max_sidelobe;
        }
            else {
        goodSignals[blockIdx.x] = 0;
        }
    }
        else {
            if (max_sidelobe < 2 && (((signal & 0b111) == 0b111) || ((signal >> (n - 3)) == 0b111))) {
                goodSignals[blockIdx.x] = signal;
                sidelobes[blockIdx.x] = max_sidelobe;
            }
            else {
                goodSignals[blockIdx.x] = 0;
            }
        }
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int n = 25;
    const size_t total_signals = 1ull << n;
    FILE* file = fopen("goodSignals.txt", "w");
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t required_mem = total_signals * (sizeof(size_t) + sizeof(int));
    if (required_mem > free_mem) {
        fprintf(stderr, "Not enough GPU memory. Required: %zu MB, Available: %zu MB\n",
            required_mem / (1024 * 1024), free_mem / (1024 * 1024));
        return 1;
    }

    size_t* d_goodSignals = nullptr;
    int* d_sidelobes = nullptr;
    cudaMalloc(&d_goodSignals, total_signals * sizeof(size_t));
    cudaMalloc(&d_sidelobes, total_signals * sizeof(int));

    cudaMemset(d_goodSignals, 0, total_signals * sizeof(size_t));

    printf("Launching %zu blocks with %d threads each...\n", total_signals, n);
    cudaEventRecord(start);
    generateAndFindAKF << <total_signals, n, n * sizeof(int) >> > (d_goodSignals, d_sidelobes, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaGetLastError();

    size_t* h_goodSignals = new size_t[total_signals];
    int* h_sidelobes = new int[total_signals];
    cudaMemcpy(h_goodSignals, d_goodSignals, total_signals * sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sidelobes, d_sidelobes, total_signals * sizeof(int), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < total_signals; i++) {
        if (h_goodSignals[i]) {
            fprintf(file, "Signal %zu: ", h_goodSignals[i]);
            for (int j = n - 1; j >= 0; j--) {
                fprintf(file, "%d", (h_goodSignals[i] >> j) & 1);
            }
            fprintf(file, " | Max sidelobe: %d\n", h_sidelobes[i]);
        }
    }
    fprintf(file, "\n Execution time: %.2f ms", ms);
    printf("\nResults saved to goodSignals.txt");
    delete[] h_goodSignals;
    delete[] h_sidelobes;
    cudaFree(d_goodSignals);
    cudaFree(d_sidelobes);
    fclose(file);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
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
