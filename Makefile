NVCC = nvcc --use_fast_math -I src -std=c++17 -O0
#-maxrregcount=0
all: oneblock multiblocks idx clock-cycle

create_results_folder:
	mkdir -p results

create_build_folder:
	mkdir -p build

oneblock: src/oneblock.cpp src/scan.cu src/scan.h create_build_folder
	$(NVCC) src/oneblock.cpp src/scan.cu -o build/oneblock.out
	
multiblocks: src/multiblocks.cpp src/scan.cu src/scan.h create_build_folder
	$(NVCC) src/multiblocks.cpp src/scan.cu -o build/multiblocks.out

idx: src/idx.cu 
	nvcc src/idx.cu -o build/idx.out

clock-cycle: src/cycle.cu
	nvcc src/cycle.cu -o build/clock-cycle.out

clean:
	rm -rf build/

json: multiblocks create_results_folder
	build/multiblocks.out | python tools/multiblocks_to_json.py -o results/multiblocks.json

analysis: multiblocks create_results_folder
	build/multiblocks.out | python tools/multiblocks_analysis.py



