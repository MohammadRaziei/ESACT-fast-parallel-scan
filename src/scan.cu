#include "scan.h"
#include "helper.cuh"

#include <stdint.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(idx) ((idx) >> LOG_NUM_BANKS + (idx) >> (2 * LOG_NUM_BANKS))





#define DEFINE_METHOD_FUNC(func) inline void func(float *dev_inout, Idx_t* dev_idx, const size_t size, const int nt)


#if SCAN_MEMORY==GLOBAL_MEMORY
#define PASS_EXTRA_VALUE(x) , x
#define GET_INDEX(i) global_indices[i]
#define PASS_INDEX_PROTOTYPE , Idx_t global_indices[]
#elif SCAN_MEMORY==GLOBAL_MEMORY2
#define PASS_EXTRA_VALUE(x) , x
#define GET_INDEX(i) global_indices->value[i]
#define PASS_INDEX_PROTOTYPE , Idx_t* global_indices
#elif SCAN_MEMORY==CONST_MEMORY
__constant__ Idx_t const_indices[1024];
#define PASS_EXTRA_VALUE(x)
#define PASS_INDEX_PROTOTYPE
#define GET_INDEX(i) const_indices[i]
#elif SCAN_MEMORY==CONST_MEMORY2
__constant__ Indices const_indices;
#define PASS_EXTRA_VALUE(x)
#define PASS_INDEX_PROTOTYPE
#define GET_INDEX(i) const_indices.value[i]
#elif SCAN_MEMORY==TEXTURE_MEMORY
texture<float, 1, cudaReadModeElementType> texture_indices;
#define PASS_EXTRA_VALUE(x)
#define PASS_INDEX_PROTOTYPE
#define GET_INDEX(i) tex1Dfetch(texture_indices, i)
#endif





//////////////// base code //////////////

__global__ void scanKernel(float *ac)
{
    extern __shared__ float sdata[]; // allocated on invocation
    float lastElement = 0;
    const int n = blockDim.x;
    const int idx = bx * n + tx;
    sdata[tx] = ac[idx];
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1)
    { // build sum in place up the tree
        __syncthreads();
        if (tx < d)
        {
            const int ai = offset * (2 * tx + 1) - 1;
            const int bi = offset * (2 * tx + 2) - 1;
            sdata[bi] += sdata[ai];
        }
        offset <<= 1;
    }
    if (tx == 0)
    {
        lastElement = sdata[n - 1];
        sdata[n - 1] = 0; // clear the last element
    }
    for (int d = 1; d < n; d <<= 1)
    { // build scan
        offset >>= 1;
        __syncthreads();
        if (tx < d)
        {
            const int ai = offset * (2 * tx + 1) - 1;
            const int bi = offset * (2 * tx + 2) - 1;
            float tmp = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += tmp;
        }
    }
    __syncthreads();
    if (tx == 0)
        ac[idx + n - 1] = lastElement;
    else
        ac[idx - 1] = sdata[tx];
}

////////////////// alg 1 ///////////////

__global__ void scanReduceIdxKernel(idx_t idx[])
{
    extern __shared__ uint32_t idata[]; // allocated on invocation
    const int n = blockDim.x;
    idata[tx] = 1;
    for (uint32_t s = n >> 1; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tx < s)
        {
            idata[tx] += idata[tx + s];
        }
    }
    if (tx == 0)
    {
        idata[0] = 0; // clear the last element
    }
    for (uint32_t s = 1; s < n; s <<= 1)
    {
        __syncthreads();
        if (tx < s)
        {
            int tmp = idata[tx + s];
            idata[tx + s] = idata[tx];
            idata[tx] += tmp;
        }
    }
    __syncthreads();
    idx[tx] = idata[tx];
}






__global__ void scanReduceKernel(float ac[] PASS_INDEX_PROTOTYPE){
    extern __shared__ float sdata[];// allocated on invocation
    float lastElement; 
    const int n = blockDim.x;
//    printf("idx: %d\n", GET_INDEX(tx));
    const int idx = bx * n + GET_INDEX(tx);
    sdata[tx] = ac[idx]; // global memory coalescing
    for (unsigned int s = n >> 1; s > 0; s >>= 1) {
        __syncthreads();
        if (tx < s) {
            sdata[tx] += sdata[tx + s];
        }
    }
    __syncthreads();
    if (tx == n - 1) {  
        lastElement = sdata[0];
        sdata[0] = 0;  // clear the last element
    }
    for (unsigned int s = 1; s < n; s <<= 1) {
        __syncthreads();
        if (tx < s) {
            float tmp = sdata[tx + s];
            sdata[tx + s] = sdata[tx];
            sdata[tx] += tmp;
        }
    }
    __syncthreads();
    if(tx == n - 1)
        ac[idx + n - 1] = lastElement;
    else 
        ac[idx-1] = sdata[tx];
}

///////// alg 2: remove a syncthread with shared memory:

__global__ void scanReduceKernel2(float ac[] PASS_INDEX_PROTOTYPE){
    extern __shared__ float sdata[];// allocated on invocation
    __shared__ float lastElement[1]; // use shared memory 
    const int n = blockDim.x;
    const int idx = bx * n + GET_INDEX(tx);
    sdata[tx] = ac[idx]; // global memory coalescing
    for (unsigned int s = n >> 1; s > 0; s >>= 1) {
        __syncthreads();
        if (tx < s) {
            sdata[tx] += sdata[tx + s];
        }
    }
    if (tx == 0) { // __syncthreads() removed
        *lastElement = sdata[0];
        sdata[0] = 0;  // clear the last element
    }
    for (unsigned int s = 1; s < n; s <<= 1) {
        __syncthreads();
        if (tx < s) {
            float tmp = sdata[tx + s];
            sdata[tx + s] = sdata[tx];
            sdata[tx] += tmp;
        }
    }
    __syncthreads();
    if(tx == n - 1)
        ac[idx + n - 1] = *lastElement;
    else 
        ac[idx-1] = sdata[tx];
}


/// alg final:


__global__ void scanReduceKernelFinal(float ac[] PASS_INDEX_PROTOTYPE){
    extern __shared__ float sdata[]; // allocated on invocation
    float lastElement;
    const int n = blockDim.x;
    const int idx = GET_INDEX(tx);
    const int offsetIdx = bx * n;
    sdata[idx] = ac[offsetIdx + tx]; 
    for (unsigned int s = n >> 1; s > 0; s >>= 1) {
        __syncthreads();
        if (tx < s) {
            sdata[tx] += sdata[tx + s];
        }
    }
    if (tx == 0) {
        lastElement = sdata[0];
        sdata[0] = 0;  // clear the last element
    }
    for (unsigned int s = 1; s < n; s <<= 1) {
        __syncthreads();
        if (tx < s) {
            float tmp = sdata[tx + s];
            sdata[tx + s] = sdata[tx];
            sdata[tx] += tmp;
        }
    }
    __syncthreads();
    if(tx == 0)
        ac[offsetIdx + n - 1] = lastElement;
    else 
        ac[offsetIdx + tx - 1] = sdata[idx];
}


/// alg padded

__global__ void scanPaddedKernel(float* g_data)
{
    extern __shared__ float s_data[];
    const int n = blockDim.x;

    int offset = 1;
    const int ai = tx;
    const int bi = tx + (n / 2);
    const int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    const int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Copy data to shared memory with padding
    s_data[ai + bankOffsetA] = (ai < n) ? g_data[bx * n + ai] : 0;
    s_data[bi + bankOffsetB] = (bi < n) ? g_data[bx * n + bi] : 0;

    // Reduction phase
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tx < d)
        {
            int ai = offset * (2 * tx + 1)- 1;
            int bi = offset * (2 * tx + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            s_data[bi] += s_data[ai];
        }
        offset <<= 1;
    }
    float savedLast = 0;
    // Post-scan phase
    if (tx == 0)
    {
        savedLast = s_data[n - 1 + bankOffsetB];
        s_data[n - 1 + bankOffsetB] = 0;
    }
    for (int d = 1; d < n; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        if (tx < d)
        {
            int ai = offset * (2 * tx + 1) - 1;
            int bi = offset * (2 * tx + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            float t = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }

    // Copy data back to global memory
    __syncthreads();
    if (ai == 0){
        g_data[n - 1] = savedLast;
    } else{
        g_data[ai - 1] = s_data[ai + bankOffsetA];
    }
    
}

//////////////////////////////////////////////////////


const std::vector<std::string> GpuScan::methodnames = {
    "BLELOCHE", 
    "PADDED",
    "SACT1",
    "SACT2",
    "ESACT",
};


DEFINE_METHOD_FUNC(runESACT)
{
    scanReduceKernelFinal<<<size/nt, nt, 
            nt*sizeof(float)>>>(dev_inout PASS_EXTRA_VALUE(dev_idx));
}

DEFINE_METHOD_FUNC(runSACT1){
    scanReduceKernel<<<size/nt, nt, 
            nt*sizeof(float)>>>(dev_inout PASS_EXTRA_VALUE(dev_idx));
}


DEFINE_METHOD_FUNC(runSACT2){
    scanReduceKernel2<<<size/nt, nt, 
            nt*sizeof(float)>>>(dev_inout PASS_EXTRA_VALUE(dev_idx));
}

DEFINE_METHOD_FUNC(runBLELOCHE){
    scanKernel<<<size/nt, nt, 
            nt*sizeof(float)>>>(dev_inout);
}

DEFINE_METHOD_FUNC(runPADDED){
        int smem_size = 2 * nt * sizeof(float) + 
                    2 * nt * sizeof(float) / NUM_BANKS + 
                    NUM_BANKS * sizeof(float);
    scanPaddedKernel<<<size/nt, nt, smem_size>>>(dev_inout);
}

#define addtomethodmap(method) {GpuScan::Methods::method, run##method}


const GpuScan::_KernelMethodsMap GpuScan::kernelMethodsMap = {
        addtomethodmap(BLELOCHE),
        addtomethodmap(PADDED),
        addtomethodmap(SACT1), 
        addtomethodmap(SACT2),
        addtomethodmap(ESACT), 
};


inline void GpuScan::runMethods(const size_t size, GpuScan::Methods method){
    callMethodsType func  = kernelMethodsMap.at(method);
    func(dev_inout, dev_idx, size, n_max_threads);
}




#define THIS_TIMER static_cast<GpuTimer*>(timer)

GpuScan::GpuScan(const size_t size, const int num_threads) : n_max_threads(num_threads) {
    timer = new GpuTimer();
//    idx_t* d_idx = reinterpret_cast<idx_t*>(dev_idx);
    malloc_dev(idx_t , dev_idx, n_max_threads);

    scanReduceIdxKernel<<<1, n_max_threads, n_max_threads*sizeof(int)>>>((idx_t*)dev_idx);
//     showDev(dev_idx, 10);

    #if SCAN_MEMORY==CONST_MEMORY
    idx_t host_idx[1024];
    dev_to_host_async(idx_t, dev_idx, host_idx, 1024);
    cudaMemcpyToSymbol(const_indices, host_idx, sizeof(idx_t) * 1024);
    #elif SCAN_MEMORY==CONST_MEMORY2
    idx_t host_idx[1024];
    dev_to_host_async(idx_t, dev_idx, host_idx, 1024);
    cudaMemcpyToSymbol(const_indices, host_idx, sizeof(idx_t) * 1024);
    #elif SCAN_MEMORY==TEXTURE_MEMORY
    // cudaBindTexture(&offset, texInput, dInput, sizeof(float)*WIDTH);

    #endif

    // showDev(dev_idx, 10);


    alloc(size);
    checkError();
}

GpuScan::GpuScan() : GpuScan(0) {}

GpuScan::~GpuScan(){
    printf("bye bye :)\n");
    cudaDeviceSynchronize();
    delete THIS_TIMER;
    cudaFree(dev_idx);
    cudaFree(dev_inout);
    cudaDeviceReset();
}
void GpuScan::alloc(const size_t size){
    if (dev_inout) 
        cudaFree(dev_inout);
    if (size > 0){
        malloc_dev(float, dev_inout, size);
    }
}

float GpuScan::getTime() const{
    return THIS_TIMER->elapsedMs();
}


void GpuScan::operator()(float* in_array, float* out_array, const size_t size, Methods method){
    run(in_array, out_array, size, method);
}


void GpuScan::operator()(float* in_array, float* out_array, const size_t size, const std::string& method){
    int m = 0;
    for(;m < methodnames.size(); ++m)
        if(methodnames[m] == method) 
            break;
    if (m >= methodnames.size()){
        printf("Unexpected method: %s\n", method.c_str());
        return;
    }
    run(in_array, out_array, size, static_cast<Methods>(m));
}


void GpuScan::run(float* in_array, float* out_array, const size_t size, const Methods method){
    host_to_dev_async(float, in_array, dev_inout, size);
    THIS_TIMER->start();
    runMethods(size, method);
//     checkError();
    THIS_TIMER->stop();
    dev_to_host(float, dev_inout, out_array, size);
}


void cpuScan(float *in_array, float *out_array, const size_t size)
{
    out_array[0] = in_array[0];
    for (size_t i = 1; i < size; ++i){
        out_array[i] = in_array[i] + out_array[i - 1];
    }
}

