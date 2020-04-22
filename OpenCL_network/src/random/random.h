//
// Created by human on 13.03.2020.
//

#ifndef OPENCL_RANDOM_H
#define OPENCL_RANDOM_H
#include <stdlib.h>
#include <math.h>
#include <float.h>
#define TWO_PI (2.0 * 3.14159265358979323846f)

void random_GaussianNoiseBoxMuller(float* return_x, float* return_y, float mu, float sigma);
#endif //OPENCL_RANDOM_H
