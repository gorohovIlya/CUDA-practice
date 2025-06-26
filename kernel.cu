#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>


__global__ void kernelAKF(unsigned int currSignal, int* akf, int n) {
    int bit_pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (bit_pos < n) {
        int my_bit = ((currSignal >> bit_pos) & 1) ? 1 : -1;
        for (int shift = 0; shift < n; shift++) {
            if (bit_pos >= shift) {
                int paired_bit_pos = bit_pos - shift;
                int paired_bit = ((currSignal >> paired_bit_pos) & 1) ? 1 : -1;
                atomicAdd(&akf[shift], my_bit * paired_bit);
            }
        }
    }
}

__global__ void generateSignals(unsigned int* signals, int n, unsigned int total_signals) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_signals) {
        signals[idx] = idx;
    }
}

int main() {
    const int N = 31;
    const unsigned int total_signals = 1u << N;

    unsigned int* d_signals;
    int* d_akf;
    cudaMalloc(&d_signals, total_signals * sizeof(unsigned int));
    cudaMalloc(&d_akf, N * sizeof(int));
    cudaMemset(d_akf, 0, N * sizeof(int));

    int blockSize = 256;
    int gridSize = (total_signals + blockSize - 1) / blockSize;
    generateSignals << <gridSize, blockSize >> > (d_signals, N, total_signals);
    cudaGetLastError();

    unsigned int first_signal;
    cudaMemcpy(&first_signal, d_signals, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    kernelAKF << <1, N >> > (first_signal, d_akf, N);
    cudaGetLastError();

    int akf[N];
    cudaMemcpy(akf, d_akf, N * sizeof(int), cudaMemcpyDeviceToHost);

    unsigned int signals[5];
    cudaMemcpy(signals, d_signals, 5 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("\nFirst 5 signals (of %u total):\n", total_signals);
    for (int i = 0; i < 5; i++) {
        printf("Signal %d: ", i);
        for (int j = N - 1; j >= 0; j--) {
            printf("%d", (signals[i] >> j) & 1);
        }
        printf("\n");
    }
    printf("\nACF of first signal:\n");
    for (int j = 0; j < N; j++) {
        printf("acf[%d] = %d\n", j, akf[j]);
    }
    cudaFree(d_signals);
    cudaFree(d_akf);

    return 0;
}
