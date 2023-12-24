#include <cuda_runtime.h>
#include "./loaderPPM/ppm.hpp"
#include "tiledGPU.hpp"
#include "utils/commonCUDA.hpp"

// Kernel CUDA pour réduire la résolution d'une image en utilisant une approche parallèle
__global__ void reduceResolutionKernel(const uint8_t* inputImage, uint8_t* outputImage, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    // Calcul des dimensions de la tuile en fonction de la résolution de sortie
    int tileWidth = inputWidth / outputWidth;
    int tileHeight = inputHeight / outputHeight;

    // Dimensions des sous-tuiles définies par les threads CUDA
    int subtileWidth = blockDim.x;
    int subtileHeight = blockDim.y;

    // Calcul de la position de départ de la tuile actuelle
    int startX = blockIdx.x * tileWidth;
    int startY = blockIdx.y * tileHeight;

    // Calcul des indices x et y du thread dans la sous-tuile
    int x = startX + threadIdx.x;
    int y = startY + threadIdx.y;

    // Variable partagée pour stocker la valeur maximale de pixel dans la tuile
    __shared__ int maxPixelValue;

    // Initialisation de la variable partagée
    if (threadIdx.x == 0 && threadIdx.y == 0)
        maxPixelValue = 0;

    // Synchronisation pour assurer que l'initialisation est terminée avant de continuer
    __syncthreads();

    // Chaque thread traite un pixel dans les sous-tuiles
    for (int i = 0; i <= tileWidth; i += subtileWidth) {
        for (int j = 0; j <= tileHeight; j += subtileHeight) {
            // Calcul des coordonnées du pixel dans l'image d'entrée
            int pixelX = x + i;
            int pixelY = y + j;

        
            if (pixelX < inputWidth && pixelY < inputHeight) {
                // Obtenir la valeur du pixel à partir de l'image d'entrée
                int pixelValue = static_cast<int>(inputImage[pixelY * inputWidth + pixelX]);

                // Mettre à jour maxPixelValue de manière atomique
                atomicMax(&maxPixelValue, pixelValue);
            }
        }
    }

    // Synchronisation pour s'assurer que tous les threads ont mis à jour maxPixelValue
    __syncthreads();

    // Calcul de la position dans outputImage pour la tuile actuelle
    int outputIndex = (startY / tileHeight) * outputWidth + (startX / tileWidth);

    // Écrire le résultat pour ce bloc
    outputImage[outputIndex] = static_cast<uint8_t>(maxPixelValue);
}


__global__ void reduceResolutionKernelNoSharedMemory(const uint8_t* inputImage, int* outputImage, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    int tileWidth = inputWidth  / outputWidth;
    int tileHeight = inputHeight / outputHeight;

    int subtileWidth = blockDim.x;
    int subtileHeight = blockDim.y;

    int startX = blockIdx.x * tileWidth;
    int startY = blockIdx.y * tileHeight;
    int outputIndex = (startY / tileHeight) * outputWidth + (startX / tileWidth);

    int x = startX + threadIdx.x;
    int y = startY + threadIdx.y;


  

    // Each thread processes one pixel in the subtiles
    for (int i = 0; i <= tileWidth; i += subtileWidth) {
        for (int j = 0; j <= tileHeight; j += subtileHeight) {
            int pixelX = x + i;
            int pixelY = y + j;

            if (pixelX < inputWidth && pixelY < inputHeight) {
                int pixelValue = static_cast<int>(inputImage[pixelY * inputWidth + pixelX]);
                atomicMax(&outputImage[outputIndex], pixelValue);
            }
        }
    }

    
}


__global__ void reduceResolutionNaiveKernel(const uint8_t* inputImage, uint8_t* outputImage, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    int tileWidth = inputWidth  / outputWidth;
    int tileHeight = inputHeight / outputHeight;

    int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    int outputY = blockIdx.y * blockDim.y + threadIdx.y;
    if (outputX < outputWidth && outputY < outputHeight) {
        int startX = outputX * tileWidth;
        int startY = outputY * tileHeight;

        int maxPixelValue = 0;
        

        for (int y = 0; y < tileHeight; ++y) {
            for (int x = 0; x < tileWidth; ++x) {
                int inputX = startX + x;
                int inputY = startY + y;

                if (inputX < inputWidth && inputY < inputHeight) {
                    int pixelValue = static_cast<int>(inputImage[inputY * inputWidth + inputX]);

                    // Update maxPixelValue without atomic operation
                    if (pixelValue > maxPixelValue) {
                        maxPixelValue = pixelValue;
                    }
                    
                }
            }
        }

        // Set the output pixel value
        outputImage[outputY * outputWidth + outputX] = static_cast<uint8_t>(maxPixelValue);
    }
}




void reduceHeightmapResolutionGPU(los::Heightmap& originalHeightmap, los::Heightmap& reducedHeightmap, int outputWidth, int outputHeight)
{
    int inputWidth = originalHeightmap.getWidth();
    int inputHeight = originalHeightmap.getHeight();
    const uint8_t *hostOriginalPixels = originalHeightmap.getPtr();
    uint8_t *hostReducedPixels = reducedHeightmap.getPtr();

    uint8_t *deviceOriginalPixels;
    uint8_t *deviceReducedPixels;

    cudaMalloc(&deviceOriginalPixels, originalHeightmap.getSize() * sizeof(uint8_t));
    cudaMalloc(&deviceReducedPixels, reducedHeightmap.getSize() * sizeof(uint8_t));
    cudaMalloc(&deviceReducedPixels, reducedHeightmap.getSize() * sizeof(uint8_t)); //for the one with no shared

    cudaMemcpy(deviceOriginalPixels, hostOriginalPixels, originalHeightmap.getSize() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    

    dim3 blockSize(16,16);
    dim3 gridSize((inputWidth + blockSize.x - 1) / blockSize.x, (inputHeight + blockSize.y - 1) / blockSize.y);

    ChronoGPU chr;
    chr.start();
    reduceResolutionNaiveKernel<<<gridSize, blockSize>>>(deviceOriginalPixels, deviceReducedPixels, inputWidth, inputHeight, outputWidth, outputHeight);
    chr.stop();
    std::cout << "====================================================================" << std::endl;
	std::cout << "          Parallel Naive Tiled Kernel version on GPU          " << std::endl;
	std::cout << "====================================================================" << std::endl;
    const float timeComputeCPU = chr.elapsedTime();
	std::cout << "-> Done : " << std::fixed << timeComputeCPU << " ms" << std::endl
			  << std::endl << std::endl;



    cudaMemcpy(hostReducedPixels, deviceReducedPixels, reducedHeightmap.getSize() * sizeof(uint8_t), cudaMemcpyDeviceToHost);

   cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Erreur après le kernel : %s\n", cudaGetErrorString(cudaStatus));
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Erreur de synchronisation du device : %s\n", cudaGetErrorString(cudaStatus));
        
    }

    // Free device memory
    cudaFree(deviceOriginalPixels);
    cudaFree(deviceReducedPixels);
}

