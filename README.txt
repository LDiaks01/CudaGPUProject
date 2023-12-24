Commande : nvcc -o test main.cu ./loaderPPM/ppm.cpp viewtestCPU.cpp  viewtestGPU.cu tiledGPU.cu tiledSequentialCPU.cpp  ./utils/chronoGPU.cu ./utils/chronoCPU.cpp

******ORGANISATION*********************************************

------------------------------------------------------
Les resultats sont stockées dans le dossier samples

------------------------------------------------------
tiledGPU.cu : Contient les fonctions pour la réduction de la taille de l'image.

            naive-tiled : __global__ void reduceResolutionNaiveKernel()
            optimized-tiled : __global__ void reduceResolutionKernel()
            void reduceHeightmapResolutionGPU sera utilisée pour paramétrer la taille des blocs et grilles et appeler le kernel.
             Il suffira juste d'appeler cette fonction dans la fonction principale.


--------------------------------------------------------
viewtest.cu : Contient les fonctions pour le calcul de la visibilité.

            naive-viewtest : determineVisibilityKernelNaive
            optimized-viewtest : determineVisibilityKernelOptimized
            void GPUViewTestCodeKernelOptimized sera appelée dans la fonction main pour lancer le kernel optimized-viewtest.
            void GPUViewTestCodeKernelNaive sera appelée dans la fonction main pour lancer le kernel naive-viewtest.


--------------------------------------------------------------------
tiledSequentialCPU.cpp : Contient la version séquentielle de la réduction reduceResolutionKernel.


--------------------------------------------------------------------------
viewtestCPU.cpp : Contient la version séquentielle du calcul de la visibilité.

--------------------------------------------------------------------------------------
main.cu : Sert à charger les heightmaps et à les passer aux autres fonctions. Pas besoin de paramétrer les tailles de grilles et de blocs dans la fonction main.





