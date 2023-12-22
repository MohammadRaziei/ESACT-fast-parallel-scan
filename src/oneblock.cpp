#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>
#undef NDEBUG
#include <assert.h>
#include <chrono>

#include "scan.h"
#include "helper.h"


int main() {
  constexpr size_t size = 1024;
  std::vector<float> in_array(size, 0);
  std::vector<float> cpuout_array(size, 0);
  std::vector<float> gpuout_array(size, 0);

  std::random_device rd; std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(1, 10);
  std::generate(in_array.begin(), in_array.end(),
                [&dis, &gen]() -> float { return dis(gen); });

  std::chrono::time_point start_time = std::chrono::steady_clock::now();
  cpuScan(in_array.data(), cpuout_array.data(), size);
  std::chrono::time_point stop_time = std::chrono::steady_clock::now();
  std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::nanoseconds>
        (stop_time - start_time).count() << std::endl;


  std::unordered_map<std::string, double> elapsedTimeMap;

  constexpr int numLoopForTime = 100'000;

  GpuScan gpuScan(size, size);

  // std::copy(methods.begin(), methods.end(), std::ostream_iterator<std::string>(std::cout, ", "));
  std::cout << std::endl;
  for (GpuScan::Methods method = GpuScan::Methods::begin; method < GpuScan::Methods::end; method = (GpuScan::Methods)((int)method+1)){
    const std::string methodstr {gpuScan.methodnames[(int)method]};
    double elapsedTime = 0;
    for (int i = 0; i < numLoopForTime; ++i){
      gpuScan(in_array.data(), gpuout_array.data(), size, method);
      elapsedTime += gpuScan.getTime();
    }
    elapsedTime /= numLoopForTime;
    elapsedTimeMap[methodstr] = elapsedTime;
    printf(">> %s: \n\ttime:\t %06.3f (us)\n", methodstr.c_str(), elapsedTime*1e3);
    const double mse = calculateMSE(cpuout_array.data(), 
                                    gpuout_array.data(), size);
    printf("\tMSE:\t %g\n", mse);
  }

  printf("\n\n========\n\n");     


  // float elapsedTime = gpuScan(in_array.data(), gpuout_array.data(), size, "warp");
  // printVec(gpuout_array);


  const double baseElapsedTime = elapsedTimeMap.at("BLELOCHE");

  for (auto it = elapsedTimeMap.begin(); it != elapsedTimeMap.end(); ++it) {
      printf("%s (%+03.2f)\t", it->first.c_str(), (baseElapsedTime / it->second) * 100 - 100);
  }

  printf("\n\n");



  return 0;
}