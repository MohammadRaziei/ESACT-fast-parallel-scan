#include "helper.cuh"

__global__ void scanReduceIdxKernel(int idx[])
{
    extern __shared__ int idata[]; // allocated on invocation
    const int n = blockDim.x;
    idata[tx] = 1;
    for (unsigned int s = n >> 1; s > 0; s >>= 1)
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
    for (unsigned int s = 1; s < n; s <<= 1)
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

int main(){
    const int m = 10;
    const int nt = 1 << m;
    int* idx;
    malloc_dev(int, idx, nt);
    printf("\n***********************\nn = %d\n\n", nt);
    scanReduceIdxKernel<<<1, nt, nt*sizeof(int)>>>(idx);
    showDev(idx, nt);
    return 0;
}
