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
    const unsigned int total_signals = 1 << N;
    int* d_akf0;
    unsigned int* d_signals;
    cudaMalloc(&d_signals, total_signals * sizeof(unsigned int));
    cudaMalloc(&d_akf0, N * sizeof(int));
    int blockSize = 256;
    int gridSize = (total_signals + blockSize) / blockSize;
    generateSignals << <gridSize, blockSize >> > (d_signals, N);
    unsigned int first_signal;
    cudaMemcpy(&first_signal, d_signals, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    kernelAKF << <1, N >> > (first_signal, d_akf0, N);
    unsigned int* signals = new unsigned int[total_signals];
    int* akf0 = new int[N];
    cudaMemcpy(akf0, d_akf0, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(signals, d_signals, total_signals * sizeof(unsigned int), cudaMemcpyDeviceToHost);
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
