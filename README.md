**Command:**
```bash
nvcc -o test main.cu ./loaderPPM/ppm.cpp viewtestCPU.cpp viewtestGPU.cu tiledGPU.cu tiledSequentialCPU.cpp ./utils/chronoGPU.cu ./utils/chronoCPU.cpp

**Organization:**

Results are stored in the "samples" folder.

**tiledGPU.cu:**
Contains functions for image resolution reduction.

- *naive-tiled:* `__global__ void reduceResolutionNaiveKernel()`
- *optimized-tiled:* `__global__ void reduceResolutionKernel()`
- `void reduceHeightmapResolutionGPU` will be used to set block and grid sizes and call the kernel. Simply call this function in the main function.

**viewtest.cu:**
Contains functions for visibility calculation.

- *naive-viewtest:* `determineVisibilityKernelNaive`
- *optimized-viewtest:* `determineVisibilityKernelOptimized`
- `void GPUViewTestCodeKernelOptimized` will be called in the main function to launch the optimized-viewtest kernel.
- `void GPUViewTestCodeKernelNaive` will be called in the main function to launch the naive-viewtest kernel.

**tiledSequentialCPU.cpp:**
Contains the sequential version of the reduceResolutionKernel.

**viewtestCPU.cpp:**
Contains the sequential version of the visibility calculation.

**main.cu:**
Used to load heightmaps and pass them to other functions. No need to set grid and block sizes in the main function.



The project entails coding four mandatory CUDA algorithms, starting with a naive approach and gradually optimizing the code for GPU parallelization. The objective is to efficiently identify visible points from a given point in the heightmap, considering the potential computational cost, especially on large-scale heightmap
