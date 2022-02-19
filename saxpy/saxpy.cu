#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);

__global__ void saxpy_kernel(int N, float alpha, float *x, float *y, float *result) {
    // Compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N)
        result[index] = alpha * x[index] + y[index];
}

void saxpyCuda(int N, float alpha, float *xarray, float *yarray, float *resultarray) {
    // why total bytes... WTF
    int totalBytes = sizeof(float) * 3 * N;

    int vectorBytes = sizeof(float) * N;

    // Compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *device_x;
    float *device_y;
    float *device_result;

    cudaMalloc((void**)&device_x, totalBytes);
    cudaMalloc((void**)&device_y, totalBytes);
    cudaMalloc((void**)&device_result, totalBytes);

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    cudaMemcpy(device_x, xarray, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, yarray, vectorBytes, cudaMemcpyHostToDevice);
    
    // run kernel
    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, device_x, device_y, device_result);
    cudaDeviceSynchronize();

    cudaMemcpy(resultarray, device_result, vectorBytes, cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));

    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);

}

void printCudaInfo() {
    // For fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
