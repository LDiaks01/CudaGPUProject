#ifndef CPUVIEWTEST_HPP
#define NAIVE_HPP

#include "loaderPPM/ppm.hpp"
#include <vector>
#include <utility>
#include "utils/chronoCPU.hpp"


bool determineVisibility(los::Heightmap& heightmap, int x1, int y1, int x2, int y2);
double calculateAngle(los::Heightmap& heightmap, int x1, int y1, int x2, int y2);
void CPUViewTestCode(los::Heightmap& heightmap, los::Heightmap& CPUHeightmap, uint32_t centre_x, u_int32_t centre_y);



#endif // CPU VIEWTEST CODE
