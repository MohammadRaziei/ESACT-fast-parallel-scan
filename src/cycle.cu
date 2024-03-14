#include "helper.cuh"
#include <stdint.h>
#include <iostream>


__global__ void clockOverheadKernel(clock_t* time){
    const clock_t startTime = clock();
    const clock_t stopTime = clock();
    *time = stopTime - startTime;
}

__global__ void scanBaseTimeKernel(clock_t* time, const int stride){
    __shared__ float sdata[2048]; // allocated on invocation
    int offset = 1;
    offset <<= stride;
    const int ai = offset * (2 * tx + 1) - 1;
    const int bi = offset * (2 * tx + 2) - 1;

    const clock_t startTime = clock();
    sdata[bi] += sdata[ai];
    const clock_t stopTime = clock();
    *time = stopTime - startTime;

}

__global__ void scanReductionTimeKernel(clock_t* time, const int stride){
    __shared__ float sdata[2048]; // allocated on invocation
    unsigned int s = 1024 >> 1;
    s >>= stride;
    const int ai = tx;
    const int bi = tx + s;

    const clock_t startTime = clock();
    sdata[ai] += sdata[bi];
    const clock_t stopTime = clock();
    *time = stopTime - startTime;
}

int main(){
    cudaDeviceReset();


    clock_t time = 0;
    clock_t* d_time;
    malloc_dev(clock_t, d_time, 1);

    clock_t overhead = 0;
    clockOverheadKernel<<<1,32>>>(d_time);
    dev_to_host(uint64_t, d_time, &overhead, 1);

    constexpr int warpsize = 32;

    std::cout << "overhead time of clock: " << overhead << std::endl;
    std::cout << "warpSize: " << warpsize << std::endl;

    GpuTimer timer;

    const int max_it = 20;
    for(int stride = 0, stopcond=1; stopcond < warpsize; stopcond <<= 1, ++stride){
        std::cout << "===========" << std::endl;
        std::cout << "Stride: " << stride << std::endl;
        std::cout << "===========" << std::endl;

        for(int i = 0; i < max_it; ++i){
            timer.start();
            scanBaseTimeKernel<<<1,32>>>(d_time, stride);
            timer.stop();
            dev_to_host(uint64_t, d_time, &time, 1);
            const float elapsed = timer.elapsedMs();
            std::cout << i << " Time: " << (time-overhead) << "\t Elapsed: " << elapsed * 1000 << " (ms)" << std::endl;
        }

        std::cout << "===========" << std::endl;

        for(int i = 0; i < max_it; ++i){
            timer.start();
            scanReductionTimeKernel<<<1,32>>>(d_time, stride);
            timer.stop();
            dev_to_host(uint64_t, d_time, &time, 1);
            const float elapsed = timer.elapsedMs();
            std::cout << i << " Time: " << (time-overhead) << "\t Elapsed: " << elapsed * 1000 << " (ms)" << std::endl;
        }
    }
    cudaFree(d_time);
    return 0;
}