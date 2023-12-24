
#ifndef __CUDA_VIEW_HPP__
#define __CUDA_VIEW_HPP__


#include "utils/commonCUDA.hpp"
#include "loaderPPM/ppm.hpp"
#include "utils/chronoGPU.hpp"

__device__ double calculateAngleGPU(const uint8_t *original_pixels, int x1, int y1, int x2, int y2, int width);
__global__ void determineVisibilityKernel2(const uint8_t* original_pixels, uint8_t* calculated_pixels, int x1, int y1, int width, int height);
void GPUViewTestCodeKernelOptimized(los::Heightmap& originalHeightmap, los::Heightmap& GPUHeightmap, int centre_x, int centre_y);
void GPUViewTestCodeKernelNaive(los::Heightmap& originalHeightmap, los::Heightmap& GPUHeightmap, int centre_x, int centre_y);
__global__ void determineVisibilityKernelOptimized(const uint8_t* original_pixels, uint8_t* calculated_pixels, int x1, int y1, int width, int height);
__device__ double calculateAngleGPU2(const uint8_t *original_pixels, int x1, int y1, int z1, int x2, int y2, int width);
#endif  //__CUDA_VIEW_HPP__