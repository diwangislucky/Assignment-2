#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);

/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// TODO: we need to implement exclusiove scan.
// we should have 2 parts:
// 1 part for up-prop
// 2nd part for down-prop
// bullshit
// we will use thread synchronization and memory barriers.
__global__ void upscan_kernel(int *device_data, int stride, int length) {
    // it's simple math,  we just calculate the stride from the thread indexes    
    // we will need to use the kernel idx.
    int arrayIdx1 = 2 * stride * (blockIdx.x * blockDim.x + threadIdx.x)
        + stride - 1;
    int arrayIdx2 = arrayIdx1 + stride;
    if (arrayIdx1 >= length || arrayIdx2 >= length) {
        return;
    }
    device_data[arrayIdx2] += device_data[arrayIdx1];
    return;
}


__global__ void downscan_kernel(int *device_data, int stride, int length) {
    int arrayIdx1 = 2 * stride * (blockIdx.x * blockDim.x + threadIdx.x)
        + stride - 1;
    int arrayIdx2 = arrayIdx1 + stride;
    if (arrayIdx1 >= length || arrayIdx2 >= length) {
        return;
    }
    int temp_val = device_data[arrayIdx2];
    device_data[arrayIdx2] += device_data[arrayIdx1];
    device_data[arrayIdx1] = temp_val;
    return;
}

// I think we can do it per thread-block.
void exclusive_scan(int *device_data, int length) {
    int rounded_length = nextPow2(length);
    // okay
    // device data.
    int stride = 1;

    // printf("up-prop\n");
    // use up-prop kernel :)
    while (2 * stride < rounded_length) {
        // TODO: how do we divide the work
        // we need to set blockIdx.x and threadIdx.x
        int threadsPerBlock = 512;
        // stride 2 * stride is the number of units
        int numThreads = (rounded_length + 2 * stride - 1) / (2 * stride);
        int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
        // printf("numThreads: %d, stride: %d, length: %d\n", 
        //         numThreads, stride, length);

        upscan_kernel<<<blocks, threadsPerBlock>>>(device_data, 
                stride, rounded_length);
        cudaDeviceSynchronize();
        stride *= 2;
    }
    // we need to be extra sure about the indices

    // cudamemset last element to 0
    // FUCK NO WONDER!!!
    // WE DID NOT DO THIS!!!
    cudaMemset(device_data + rounded_length - 1, 0, sizeof(int)); 

    // printf("down-prop\n");
    // use down-prop  kernel
    while (stride > 0) {
        // TODO: how do we divide the work
        int threadsPerBlock = 512;
        int numThreads = (rounded_length + 2 * stride - 1) / (2 * stride);
        int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
        // printf("numThreads: %d, stride: %d, length: %d\n", 
        //         numThreads, stride, length);
        downscan_kernel<<<blocks, threadsPerBlock>>>(device_data, stride,
                rounded_length);
        cudaDeviceSynchronize();
        stride /= 2;
    }
    return;
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int *inarray, int *end, int *resultarray) {
    int *device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_data, end - inarray);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);
    return overallDuration;
}


/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int *inarray, int *end, int *resultarray) {
    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void is_peak(int *device_data, int *result, int length) {
    int arrayIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // specification
    if (arrayIdx < 0 || arrayIdx >= length) {
        return;
    }

    if (arrayIdx == 0 || arrayIdx == length - 1) {
        result[arrayIdx] = 0;
        return;
    }

    if (device_data[arrayIdx - 1] < device_data[arrayIdx] &&
            device_data[arrayIdx + 1] < device_data[arrayIdx]) {
        result[arrayIdx] = 1;
        return;
    }
    result[arrayIdx] = 0;

    return;
}

__global__ void reverse_peak_mapping(int *input, int *result, int length) {
    // input is reverse mapping.
    int arrayIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (arrayIdx <= 0 || arrayIdx >= length - 1) {
        return;
    }

    if (input[arrayIdx] < input[arrayIdx + 1]) {
        result[input[arrayIdx]] = arrayIdx;
    }

    return;
    // specification
}

int find_peaks(int *device_input, int length, int *device_output) {
    int roundedLength = nextPow2(length);

    int *isPeaksArray;
    cudaMalloc((void **)&isPeaksArray, roundedLength * sizeof(int));


    int threadsPerBlock = 512;
    int numThreads = roundedLength;
    int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
    
    // find_peaks.
    is_peak<<<blocks, threadsPerBlock>>>(device_input, isPeaksArray, length);

    cudaDeviceSynchronize();

    // scan booleans  to understand where the peaks are
    exclusive_scan(isPeaksArray, length);
    reverse_peak_mapping<<<blocks, threadsPerBlock>>>(isPeaksArray,
            device_output, length);
    cudaDeviceSynchronize();
    int size;
    //  this is the fault
    // cudaMemcpy(&size, (isPeaksArray + length - 1), 
    //         sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&size, (isPeaksArray + length - 1), sizeof(int), 
            cudaMemcpyDeviceToHost);
    cudaFree(isPeaksArray);

    // TODO: we need to reverse map the integers
    return size;
}

/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
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
