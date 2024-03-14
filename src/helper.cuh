#ifndef HELPER_CUH
#define HELPER_CUH

#include <stdio.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may define other parameters here!
// you may define other macros here!
#define malloc_dev(TYPE, dev_var, len) \
  cudaMalloc((void**)&(dev_var), (len) * sizeof(TYPE))
#define dev_to_host(TYPE, dev_var, host_var, len) \
  cudaMemcpy(host_var, dev_var, (len) * sizeof(TYPE), cudaMemcpyDeviceToHost)
#define dev_to_host_async(TYPE, dev_var, host_var, len)    \
  cudaMemcpyAsync(host_var, dev_var, (len) * sizeof(TYPE), cudaMemcpyDeviceToHost)
#define host_to_dev(TYPE, host_var, dev_var, len) \
  cudaMemcpy(dev_var, host_var, (len) * sizeof(TYPE), cudaMemcpyHostToDevice)
#define host_to_dev_async(TYPE, host_var, dev_var, len)    \
  cudaMemcpyAsync(dev_var, host_var, (len) * sizeof(TYPE), cudaMemcpyHostToDevice)
  
#define showDev(inp, n) cudaDeviceSynchronize(); printf("%s: ", #inp); \
  printKernel<<<1,1>>>(inp, n); cudaDeviceSynchronize(); printf("\n")

#define _FAKE_TEMPLATE_ template<typename=void>

// you may define other functions here!
_FAKE_TEMPLATE_
__host__ __device__ void print(const float a[], const int n){
    for (int i = 0; i < n; ++i) printf("%g, ", a[i]);
    printf("\n");
}

template<typename T>
__host__ __device__ void print(const T a[], const int n){
    for (int i = 0; i < n; ++i) printf("%i, ", a[i]);
    printf("\n");
}
template<typename T>
__global__ void printKernel(const T a[], const int n){
    print(a, n);
}

#define checkError()       \
  cudaDeviceSynchronize(); \
  printError(cudaGetLastError(), __LINE__)
inline void printError(const cudaError& status, const int& line,
                       bool abort = true) {
  if (status != cudaSuccess) {
    printf("\n[ERROR] in line: %i\t message: \"%s\"\n", line,
           cudaGetErrorString(status));
    if (abort) exit(status);
  }
}


struct GpuTimer{
	cudaEvent_t _start, _stop;
	GpuTimer(){
		cudaEventCreate(&_start);
		cudaEventCreate(&_stop);
	}
	~GpuTimer(){
		cudaEventDestroy(_start);
		cudaEventDestroy(_stop);
	}
	void start(){
		cudaEventRecord(_start, 0);
	}
	void stop(){
		cudaEventRecord(_stop, 0);
	}
	float elapsedMs(){
		float elapsed;
		cudaEventSynchronize(_stop);
		cudaEventElapsedTime(&elapsed, _start, _stop);
		return elapsed;
	}
};

#endif // HELPER_CUH