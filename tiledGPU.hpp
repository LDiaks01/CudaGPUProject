#include "utils/commonCUDA.hpp"
#include "loaderPPM/ppm.hpp"
#include "utils/chronoGPU.hpp"


__global__ void reduceResolutionNaiveKernel(const uint8_t* inputImage, uint8_t* outputImage, int inputWidth, int inputHeight, int outputWidth, int outputHeight);
__global__ void reduceResolutionKernel(const uint8_t* inputImage, uint8_t* outputImage, int inputWidth, int inputHeight, int outputWidth, int outputHeight);
__global__ void reduceResolutionKernelNoSharedMemory(const uint8_t* inputImage, int* outputImage, int inputWidth, int inputHeight, int outputWidth, int outputHeight);
void reduceHeightmapResolutionGPU(los::Heightmap& originalHeightmap, los::Heightmap& reducedHeightmap, int outputWidth, int outputHeight);