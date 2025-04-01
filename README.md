# ESACT: Efficient Sequentially-Addressed Calculating Tree for Bank-Conflict-Free Parallel Prefix Sum on GPUs  
**Mohammad Raziei**  
*Sharif University of Technology, Tehran, Iran*  

---

## 📌 Abstract  
This repository presents **ESACT**, a novel GPU-optimized parallel prefix-sum (scan) algorithm that eliminates shared memory bank conflicts without requiring memory padding. Built upon the Blelloch algorithm, ESACT employs mathematically derived index reordering to transform interleaved memory access into sequential patterns, achieving **27% faster computation** compared to traditional methods. The approach ensures coalesced global memory access, supports both inclusive/exclusive scans, and generalizes efficiently to arbitrary input sizes.  

---

## 🚀 Key Features  
- **Bank-Conflict Elimination**: Sequential addressing via precomputed indices (`Ωₙ`) removes shared memory bank conflicts, improving parallelism.  
- **No Shared Memory Padding**: Reduces memory overhead while maintaining performance.  
- **Coalesced Global Memory Access**: Optimized data loading/storing patterns leverage GPU caching.  
- **Flexibility**: Compatible with existing optimizations and applicable to diverse Abelian group operations (e.g., `+`, `max`, `min`).  
- **Visualization Tools**: Interactive diagrams for Blelloch/ESACT trees, permutation matrices, and index mappings.  

---

## 📊 Performance Highlights  
- **27.24% speedup** over the vanilla Blelloch algorithm (single-block, NVIDIA RTX 3060).  
- Outperforms padded Blelloch variants across multiple GPUs (RTX 4060, A100, T4, etc.).  
- Minimal synchronization overhead with optimized CUDA kernels.  

![Benchmark Results](https://github.com/MohammadRaziei/ESACT-fast-parallel-scan/raw/results/NVIDIA-GeForce-RTX-3090/multiblocks_analysis_results_rel.png)  
*Relative execution times of ESACT vs. baseline methods on NVIDIA GPUs.*  

---

## 🛠️ Implementation Details  
### Code Structure  
- **CUDA/C++**: Kernels for Blelloch, SACT, and ESACT algorithms.  
- **Makefile**: Compiles code for NVIDIA GPUs (tested with CUDA 12+).  
- **Precomputed Indices**: `Ωₙ` tables for `n = 1, 2, ..., 1024`.  

### Build & Run  
```bash  
git clone https://github.com/MohammadRaziei/ESACT-fast-parallel-scan  
cd ESACT-fast-parallel-scan  
make clean && make  
./bin/scan --size 1024  # Example: Run scan for 1024 elements  
```  

---

## 📈 Visualization & Analysis  
Explore interactive diagrams and matrices:  
- **Blelloch vs. ESACT Trees**: [Calculation Tree Comparison](https://mohammadraziei.github.io/ESACT-fast-parallel-scan/#tree)  
- **Permutation Matrices**: [Index Reordering Patterns](https://mohammadraziei.github.io/ESACT-fast-parallel-scan/#matrix)  
- **Bank Conflict Analysis**: [Shared Memory Access Patterns](https://mohammadraziei.github.io/ESACT-fast-parallel-scan/#bank)  

---

## 📖 Citation  
If you use ESACT in your research, please cite:  
```bibtex  
@article{raziei2024esact,  
  title={ESACT: An Index-Reordering Approach to Eliminate Bank Conflicts in Parallel Prefix-Sum Algorithms without Shared Memory Padding},  
  author={Raziei, Mohammad and Kazemi, Reza and Hashemi, Matin},  
  journal={Journal of Parallel and Distributed Computing},  
  year={2024}  
}  
```  

---

## 🔗 Links  
- **GitHub Repo**: [Code & Documentation](https://github.com/MohammadRaziei/ESACT-fast-parallel-scan)  
- **Interactive Visualizations**: [ESACT Explorer](https://mohammadraziei.github.io/ESACT-fast-parallel-scan/)  
- **Contact**: [mohammadraziei@ee.sharif.edu](mailto:mohammadraziei@ee.sharif.edu)  

---

## 📜 License  
This project is licensed under the MIT License. See [LICENSE](https://github.com/MohammadRaziei/ESACT-fast-parallel-scan/blob/main/LICENSE) for details.
