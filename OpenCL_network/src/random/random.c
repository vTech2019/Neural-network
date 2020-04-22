//
// Created by human on 13.03.2020.
//
#include "random.h"
#define TWO_PI (2.0 * 3.14159265358979323846f)

void random_GaussianNoiseBoxMuller(float* return_x, float* return_y, float mu, float sigma)
{
    float u1, u2;
    float z0, z1;
    do
    {
        u1 = rand() * (1.0f / RAND_MAX);
        u2 = rand() * (1.0f / RAND_MAX);
    }while (u1 <= FLT_MIN);
    z0 = sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI * u2);
    z1 = sqrtf(-2.0f * logf(u1)) * sinf(TWO_PI * u2);
    *return_y = z1 * sigma + mu;
    *return_x = z0 * sigma + mu;
}

#undef TWO_PI
