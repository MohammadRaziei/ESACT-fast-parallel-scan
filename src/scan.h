#ifndef MY_KERNELS_H
#define MY_KERNELS_H

#include <unordered_map>
#include <string>
#include <vector>


#define GLOBAL_MEMORY 1
#define GLOBAL_MEMORY2 2
#define CONST_MEMORY 3
#define CONST_MEMORY2 4
#define TEXTURE_MEMORY 5




#ifndef SCAN_MEMORY
#define SCAN_MEMORY GLOBAL_MEMORY
#endif

#if SCAN_MEMORY==GLOBAL_MEMORY
#pragma message("SCAN_MEMORY==GLOBAL_MEMORY")
#elif SCAN_MEMORY==CONST_MEMORY
#pragma message("SCAN_MEMORY==CONST_MEMORY")
#elif SCAN_MEMORY==CONST_MEMORY2
#pragma message("SCAN_MEMORY==CONST_MEMORY2")
#endif

typedef uint16_t idx_t;
struct Indices{idx_t value[1024];};
#if SCAN_MEMORY==GLOBAL_MEMORY2
using Idx_t = Indices;
#else
using Idx_t = idx_t;
#endif

void cpuScan(float *in_array, float *out_array, const size_t size);

class GpuScan {
public:
    enum class Methods {
        begin = 0,
        BLELOCHE = begin,
        PADDED,
        SACT1,
        SACT2,
        ESACT,
        end
    };

    typedef void (*callMethodsType)(float *, Idx_t *, const size_t, const int);

    const static std::vector<std::string> methodnames;
    typedef std::unordered_map<Methods, callMethodsType> _KernelMethodsMap;
    const static _KernelMethodsMap kernelMethodsMap;


public:
    GpuScan();

    GpuScan(const size_t size, const int num_threads = 1024);

    ~GpuScan();

    void alloc(const size_t size);

    void operator()(float *in_array, float *out_array, const size_t size, Methods method = Methods::ESACT);

    void operator()(float *in_array, float *out_array, const size_t size, const std::string &method);

    void run(float *in_array, float *out_array, const size_t size, const Methods method);

    float getTime() const;

    inline void runMethods(const size_t size, Methods method);

private:

    const int n_max_threads;
    float *dev_inout = nullptr;
    Idx_t *dev_idx = nullptr;
    void *timer;
};

#endif /// MY_KERNELS_H