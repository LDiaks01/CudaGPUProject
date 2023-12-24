#include <iostream>
#include <vector>
#include <cmath>
#include "tiledSequentialCPU.hpp"



void reduceResolutionCPU(const uint8_t* inputImage, uint8_t* outputImage, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    int tileWidth = (inputWidth + outputWidth - 1) / outputWidth;
    int tileHeight = (inputHeight + outputHeight - 1) / outputHeight;

    ChronoCPU chr;
    chr.start();
    
    for (int outputY = 0; outputY < outputHeight; ++outputY) {
        for (int outputX = 0; outputX < outputWidth; ++outputX) {
            int startX = outputX * tileWidth;
            int startY = outputY * tileHeight;

            u_int8_t maxPixelValue = 0;

            for (int y = 0; y < tileHeight; ++y) {
                for (int x = 0; x < tileWidth; ++x) {
                    int inputX = startX + x;
                    int inputY = startY + y;

                    // Verification pour ne pas d2border
                    if (inputX < inputWidth && inputY < inputHeight) {
                        uint8_t pixelValue = static_cast<u_int8_t>(inputImage[inputY * inputWidth + inputX]);

                        // Update maxPixelValue
                        if (pixelValue > maxPixelValue) {
                            maxPixelValue = pixelValue;
                        }
                    }
                }
            }

            //Image resultat
            outputImage[outputY * outputWidth + outputX] = static_cast<uint8_t>(maxPixelValue);
        }
    }

    chr.stop();
    std::cout << "============================================" << std::endl;
	std::cout << "         Sequential TILED  version on CPU          " << std::endl;
	std::cout << "============================================" << std::endl;

    const float timeComputeCPU = chr.elapsedTime();
	std::cout << "-> Done : " << std::fixed << timeComputeCPU << " ms" << std::endl
			  << std::endl << std::endl;

}
