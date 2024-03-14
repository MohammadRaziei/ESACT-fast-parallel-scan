#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>
#undef NDEBUG
#include <assert.h>
#include <chrono>
#include <stdio.h>



#include "scan.h"
#include "helper.h"

typedef std::chrono::steady_clock chrono_clock;


int main() {
	std::random_device rd; std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(1, 10);

	constexpr size_t nt = 1024;
	constexpr size_t n_repeat = 100;
	std::chrono::time_point<chrono_clock> start_time, stop_time;
	double tm = 0;


	for (size_t m = 10; m < 33; ++m) {
		printf("\n\n=========================\nm = %zd\n=========================\n", m);
		const size_t size = 1ll << m;
		std::vector<float> in_array(size, 0);
		std::vector<float> cpuout_array(size, 0);
		std::vector<float> gpuout_array(size, 0);


		std::generate(in_array.begin(), in_array.end(),
			[&dis, &gen]() -> float { return /*dis(gen)*/1; });

		tm = 0;
		for (size_t r = 0; r < n_repeat; ++r) {
			start_time = chrono_clock::now();
			cpuScan(in_array.data(), cpuout_array.data(), size);
			stop_time = chrono_clock::now();
			tm = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() / 1000.;
    		printf("CPU time: %.3f ms\n", tm );
		}

		GpuScan gpuScan(size);

		for (GpuScan::Methods method = GpuScan::Methods::begin; method < GpuScan::Methods::end; method = (GpuScan::Methods)((int)method + 1)) {

			const std::string methodstr{ gpuScan.methodnames[(int)method] };

			printf("\nMethod: %s\n", methodstr.c_str());

			tm = 0;
			double ktm = 0;
			for (size_t r = 0; r < n_repeat; ++r) {
				fill(gpuout_array.begin(), gpuout_array.end(), 0);
				start_time = chrono_clock::now();
				gpuScan(in_array.data(), gpuout_array.data(), size, method);
				stop_time = chrono_clock::now();
				tm = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time - start_time).count() * 1e-6;
				ktm = gpuScan.getTime();
        		printf("Total Time: %.6f ms\n", tm);
			    printf("Kernel Time: %.6f ms\n", ktm);
            }

			const double mse = calculateMSE(cpuout_array.data(), gpuout_array.data(), nt);
			if (mse > 1e-4) 
				throw std::runtime_error("mse is too large");
			printf("MSE: %g\n", mse);
		}
	}
	
	return 0;
}
