#include "viewtestGPU.hpp"
#include "./loaderPPM/ppm.hpp"
#include "utils/commonCUDA.hpp"

// Kernel CUDA : Détermine la visibilité des points par rapport à une ligne dans une Heightmap de manière naïve.
__global__ void determineVisibilityKernelNaive(const uint8_t* original_pixels, uint8_t* calculated_pixels, int x1, int y1, int width, int height) {
    // Itération sur les blocs et les threads CUDA pour traiter les pixels de la Heightmap
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = x + y * width;

    int length = abs(x - x1) > abs(y - y1) ? abs(x - x1) : abs(y - y1);

    // Calcul des pas horizontaux et verticaux.
    double dx = static_cast<double>(x - x1) / length;
    double dy = static_cast<double>(y - y1) / length;

    // Initialise les coordonnées pour l'arrondissement en int pour la rasterisation de la ligne.
    double x_dda = x1 + dx;
    double y_dda = y1 + dy;

    bool isVisible = true;

    for (int i = 1; i < length; ++i) {
        // Vérifie si l'angle entre c et le point actuel de la ligne est supérieur à l'angle entre c et le point p1.
        if (calculateAngleGPU(original_pixels, x1, y1, x_dda, y_dda, width) >= calculateAngleGPU(original_pixels, x1, y1, x, y, width)) {
            isVisible = false;
            break;
        }

        // Mise à jour des coordonnées pour le prochain point de la droite.
        x_dda += dx;
        y_dda += dy;
    }

    if (isVisible) {
        calculated_pixels[tid] = 255;
    }
}

// Kernel CUDA : Détermine la visibilité des points par rapport à une ligne dans une Heightmap de manière optimisée.
__global__ void determineVisibilityKernelOptimized(const uint8_t* original_pixels, uint8_t* calculated_pixels, int x1, int y1, int width, int height) {
    // Itération sur les blocs et les threads CUDA pour traiter les pixels de la Heightmap
    for (int x = threadIdx.x + blockIdx.x * blockDim.x; x < width; x += blockDim.x * gridDim.x) {
        for (int y = threadIdx.y + blockIdx.y * blockDim.y; y < height; y += blockDim.y * gridDim.y) {
            int tid = x + y * width;

            int length = abs(x - x1) > abs(y - y1) ? abs(x - x1) : abs(y - y1);

            // Calcul des pas horizontaux et verticaux.
            double dx = static_cast<double>(x - x1) / length;
            double dy = static_cast<double>(y - y1) / length;

            // Initialise les coordonnées pour l'arrondissement en int pour la rasterisation de la ligne.
            double x_dda = x1 + dx;
            double y_dda = y1 + dy;

            bool isVisible = true;

            double z1 = original_pixels[x1 + y1 * width];
            // Calculer et stocker l'angle
            double angle_p1 = calculateAngleGPU2(original_pixels, x1, y1, z1, x, y, width);

            for (int i = 1; i < length; ++i) {
                // Vérifie si l'angle entre c et le point actuel de la ligne est supérieur à l'angle entre c et le point p1.
                if (calculateAngleGPU2(original_pixels, x1, y1, z1, x_dda, y_dda, width) >= angle_p1) {
                    isVisible = false;
                    break;
                }

                // Mise à jour des coordonnées pour le prochain point de la droite.
                x_dda += dx;
                y_dda += dy;
            }

            if (isVisible) {
                calculated_pixels[tid] = 255;
            }
        }
    }
}
__device__ double calculateAngleGPU(const uint8_t *original_pixels, int x1, int y1, int x2, int y2, int width)
{
    // Récupère les valeurs des pixels à (x1, y1) et (x2, y2).
    double z1 = static_cast<double>(original_pixels[ x1 + y1 * width ]);
    double z2 = static_cast<double>(original_pixels[ x2 + y2 * width ]);

    // Calcul des différences en x et y.
    double dx = static_cast<double>(x2 - x1);
    double dy = static_cast<double>(y2 - y1);

    // Calcul de l'angle en radians en utilisant l'arc tangente.
    double angle = std::atan2(z2 - z1, std::sqrt(dx * dx + dy * dy));

    // Renvoie l'angle calculé.
    return angle;
}

__device__ double calculateAngleGPU2(const uint8_t *original_pixels, int x1, int y1, int z1, int x2, int y2, int width)
{
    // Récupère les valeurs des pixels à (x1, y1) et (x2, y2).
    double z2 = static_cast<double>(original_pixels[ x2 + y2 * width ]);

    // Calcul des différences en x et y.
    double dx = static_cast<double>(x2 - x1);
    double dy = static_cast<double>(y2 - y1);

    // Calcul de l'angle en radians en utilisant l'arc tangente.
    double angle = std::atan2(z2 - z1, std::sqrt(dx * dx + dy * dy));

    // Renvoie l'angle calculé.
    return angle;
}

// Fonction CUDA : Lance le kernel naïf pour le test de visibilité sur GPU.
void GPUViewTestCodeKernelNaive(los::Heightmap& originalHeightmap, los::Heightmap& GPUHeightmap, int centre_x, int centre_y) {
    // Obtient les informations sur la taille de la Heightmap.
    int largeur = originalHeightmap.getWidth();
    int hauteur = originalHeightmap.getHeight();

    // Obtient les pointeurs vers les pixels de la Heightmap.
    const uint8_t* hostOriginalPixels = originalHeightmap.getPtr();
    uint8_t* hostCalculatedPixels = GPUHeightmap.getPtr();

    // Déclare et alloue l'espace mémoire sur le GPU.
    uint8_t* deviceOriginalPixels;
    uint8_t* deviceCalculatedPixels;

    HANDLE_ERROR(cudaMalloc(&deviceOriginalPixels, originalHeightmap.getSize() * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc(&deviceCalculatedPixels, originalHeightmap.getSize() * sizeof(uint8_t)));

    // Copie les données depuis le CPU vers le GPU.
    HANDLE_ERROR(cudaMemcpy(deviceOriginalPixels, hostOriginalPixels, originalHeightmap.getSize() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(deviceCalculatedPixels, hostCalculatedPixels, originalHeightmap.getSize() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Configure les dimensions des blocs et des grilles pour le kernel.
    dim3 blockSize(32, 32);
    dim3 gridSize((largeur + blockSize.x - 1) / blockSize.x, (hauteur + blockSize.y - 1) / blockSize.y);

    // Chronomètre le temps d'exécution du kernel naïf.
    ChronoGPU chr;
    chr.start();
    determineVisibilityKernelNaive<<<gridSize, blockSize>>>(deviceOriginalPixels, deviceCalculatedPixels, centre_x, centre_y, largeur, hauteur);
    chr.stop();

    // Affiche les résultats.
    std::cout << "====================================================================" << std::endl;
    std::cout << "          Version parallèle naïve du kernel ViewTest sur GPU          " << std::endl;
    std::cout << "====================================================================" << std::endl;
    const float timeComputeCPU = chr.elapsedTime();
    std::cout << "-> Done : " << std::fixed << timeComputeCPU << " ms" << std::endl << std::endl << std::endl;

    // Vérifie les erreurs CUDA après l'exécution du kernel.
    cudaError_t cuda_err2 = cudaGetLastError();
    if (cuda_err2 != cudaSuccess) {
        fprintf(stderr, "Erreur CUDA : %s (%s)\n", cudaGetErrorString(cuda_err2), cudaGetErrorName(cuda_err2));
    }

    // Copie les résultats depuis le GPU vers le CPU.
    HANDLE_ERROR(cudaMemcpy(hostCalculatedPixels, deviceCalculatedPixels, originalHeightmap.getSize() * sizeof(uint8_t),  cudaMemcpyDeviceToHost));

    // Libère l'espace mémoire sur le GPU.
    cudaFree(deviceOriginalPixels);
    cudaFree(deviceCalculatedPixels);
}

// Fonction CUDA : Lance le kernel optimisé pour le test de visibilité sur GPU.
void GPUViewTestCodeKernelOptimized(los::Heightmap& originalHeightmap, los::Heightmap& GPUHeightmap, int centre_x, int centre_y) {
    // Obtient les informations sur la taille de la Heightmap.
    int largeur = originalHeightmap.getWidth();
    int hauteur = originalHeightmap.getHeight();

    // Obtient les pointeurs vers les pixels de la Heightmap.
    const uint8_t* hostOriginalPixels = originalHeightmap.getPtr();
    uint8_t* hostCalculatedPixels = GPUHeightmap.getPtr();

    // Déclare et alloue l'espace mémoire sur le GPU.
    uint8_t* deviceOriginalPixels;
    uint8_t* deviceCalculatedPixels;

    ChronoGPU chr;
    chr.start();
    HANDLE_ERROR(cudaMalloc(&deviceOriginalPixels, originalHeightmap.getSize() * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMalloc(&deviceCalculatedPixels, originalHeightmap.getSize() * sizeof(uint8_t)));

    // Copie les données depuis le CPU vers le GPU.
    HANDLE_ERROR(cudaMemcpy(deviceOriginalPixels, hostOriginalPixels, originalHeightmap.getSize() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(deviceCalculatedPixels, hostCalculatedPixels, originalHeightmap.getSize() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Configure les dimensions des blocs et des grilles pour le kernel.
    dim3 blockSize(8, 8);
    dim3 gridSize((largeur + blockSize.x - 1) / blockSize.x, (hauteur + blockSize.y - 1) / blockSize.y);

    // Chronomètre le temps d'exécution du kernel optimisé.
    determineVisibilityKernelOptimized<<<gridSize, blockSize, 1024 * sizeof(int)>>>(deviceOriginalPixels, deviceCalculatedPixels, centre_x, centre_y, largeur, hauteur);
    chr.stop();

    // Affiche les résultats.
    std::cout << "====================================================================" << std::endl;
    std::cout << "          Version parallèle optimisée du kernel ViewTest sur GPU          " << std::endl;
    std::cout << "====================================================================" << std::endl;
    const float timeComputeCPU = chr.elapsedTime();
    std::cout << "-> Done : " << std::fixed << timeComputeCPU << " ms" << std::endl << std::endl << std::endl;

    // Vérifie les erreurs CUDA après l'exécution du kernel.
    cudaError_t cuda_err1 = cudaGetLastError();
    if (cuda_err1 != cudaSuccess) {
        fprintf(stderr, "Erreur CUDA : %s (%s)\n", cudaGetErrorString(cuda_err1), cudaGetErrorName(cuda_err1));
    }

    // Copie les résultats depuis le GPU vers le CPU.
    HANDLE_ERROR(cudaMemcpy(hostCalculatedPixels, deviceCalculatedPixels, originalHeightmap.getSize() * sizeof(uint8_t),  cudaMemcpyDeviceToHost));

    // Libère l'espace mémoire sur le GPU.
    cudaFree(deviceOriginalPixels);
    cudaFree(deviceCalculatedPixels);
}

