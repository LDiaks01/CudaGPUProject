#include "viewtestCPU.hpp"
#include "loaderPPM/ppm.hpp"
#include "utils/commonCUDA.hpp"
#include "viewtestGPU.hpp"
#include <vector>
#include <utility>
#include "tiledGPU.hpp"
#include "tiledSequentialCPU.hpp"


using std::vector;
using std::pair;



int main()
{
    los::Heightmap heightmap("samples/1.input.ppm");
    //los::Heightmap heightmap("samples/limousin-full.ppm");
    
    // Caractéristiques de la HeightMap
    uint32_t largeur = heightmap.getWidth();
    uint32_t hauteur = heightmap.getHeight();
    printf("Largeur et hauteur de la HeightMap, %d, %d \n", largeur, hauteur);
    // Point choisi comme centre
    uint32_t centre_x = 245;
    uint32_t centre_y = 497;
    
    /* 
    //Lancement de la version CPU viewtest
    los::Heightmap CPUHeightmap(largeur, hauteur);
    CPUViewTestCode(heightmap, CPUHeightmap, centre_x, centre_y);
    CPUHeightmap.saveTo("./samples/CPUNaive");
    */
    

    
    //GPU CODE pour viewtest
    los::Heightmap GPUHeightmap(largeur, hauteur);
    GPUViewTestCodeKernelOptimized(heightmap, GPUHeightmap, centre_x, centre_y); // kernel optimized
    //GPUViewTestCodeKernelNaive(heightmap, GPUHeightmap, centre_x, centre_y); // kernel naive
    GPUHeightmap.saveTo("./samples/GPU_result");
    

    //tiled part

    int outputWidth =  10;// specify the desired width of the reduced heightmap;
    int outputHeight = 10;// specify the desired height of the reduced heightmap;

    // Create a new Heightmap object for the reduced heightmap
    los::Heightmap reducedHeightmap(outputWidth, outputHeight);

    // Call the CUDA function to reduce the heightmap resolution / changer dans la fonction pour basculer entre naive et optimized
    reduceHeightmapResolutionGPU(heightmap, reducedHeightmap, outputWidth, outputHeight);
    //reduceResolutionCPU(heightmap.getPtr(), reducedHeightmap.getPtr(), largeur, hauteur, outputHeight, outputWidth); // sur CPU
    
    reducedHeightmap.saveTo("./samples/tiled_result");

}
 

