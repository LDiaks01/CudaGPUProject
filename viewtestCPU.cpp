#include "viewtestCPU.hpp"
#include <iostream>
#include <vector>
#include <cmath>

// Cette fonction détermine la visibilité d'un point par rapport à une ligne dans une Heightmap.
// x1, y1 : Coordonnées du point de vue c (centre).
// x2, y2 : Coordonnées du point de destination p1 de la ligne.
bool determineVisibility(los::Heightmap& heightmap, int x1, int y1, int x2, int y2) {
    bool isVisible = true;

    // Calcule la longueur de la ligne.
    int length = std::max(abs(x2 - x1), abs(y2 - y1));

    // Calcul des pas horizontaux et verticaux.
    double dx = static_cast<double>(x2 - x1) / length;
    double dy = static_cast<double>(y2 - y1) / length;

    // Initialise les coordonnées pour l'arrondissement en int pour la rasterisation de la ligne.
    double x = x1 + dx;
    double y = y1 + dy;

    // Parcours des points de la ligne.
    for (int i = 1; i < length; ++i) {
        // Vérifie si l'angle entre c et le point actuel de la ligne est supérieur à l'angle entre c et le point p1.
        if(calculateAngle(heightmap, x1, y1, x, y) >= calculateAngle(heightmap, x1, y1, x2, y2)) {
            // Si l'angle est supérieur ou égal, la visibilité de ce point est fausse, et la boucle est interrompue.
            isVisible = false;
            break;
        }

        // Mise à jour des coordonnées pour le prochain point de la droite.
        x += dx;
        y += dy;
    }

    return isVisible;
}

// x1, y1 : Coordonnées du premier point. / ici toujours le centre
// x2, y2 : Coordonnées du deuxième point.
double calculateAngle(los::Heightmap& heightmap, int x1, int y1, int x2, int y2) {
    // Récupère les valeurs des pixels à (x1, y1) et (x2, y2).
    double z1 = static_cast<double>(heightmap.getPixel(x1, y1));
    double z2 = static_cast<double>(heightmap.getPixel(x2, y2));

    // Calcul des différences en x et y.
    double dx = static_cast<double>(x2 - x1);
    double dy = static_cast<double>(y2 - y1);

    // Calcul de l'angle en radians en utilisant l'arc tangente.
    double angle = std::atan2(z2 - z1, std::sqrt(dx * dx + dy * dy));

    // Renvoie l'angle calculé.
    return angle;
}

// Fonction de lancement pour le code du CPU
void CPUViewTestCode(los::Heightmap& heightmap, los::Heightmap& CPUHeightmap, uint32_t centre_x, uint32_t centre_y) {
    ChronoCPU chr;

    uint32_t largeur = heightmap.getWidth();
    uint32_t hauteur = heightmap.getHeight();
    chr.start(); 
    for(int i = 0; i < largeur; i++) {
        for(int j = 0; j < hauteur; j++) {
            if(determineVisibility(heightmap, centre_x, centre_y, i, j)) {
                CPUHeightmap.setPixel(i, j, 255);
            }
        }
    }
    chr.stop();

    std::cout << "============================================" << std::endl;
    std::cout << "         Version séquentielle de ViewTest sur CPU          " << std::endl;
    std::cout << "============================================" << std::endl;

    const float timeComputeCPU = chr.elapsedTime();
    std::cout << "-> Terminé : " << std::fixed << timeComputeCPU << " ms" << std::endl
              << std::endl << std::endl;
}




