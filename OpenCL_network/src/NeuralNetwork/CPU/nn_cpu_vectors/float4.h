//
// Created by human on 30.03.2020.
//

#ifndef OPENCL_FLOAT4_H
#define OPENCL_FLOAT4_H
#include <stdalign.h>

struct float4;
typedef struct float4 float4;
struct float4{
    alignas(16) float s[4];
};

static inline float4 float4_add(const float4 vector_0, const float4 vector_1) {
    const float4 result = { vector_0.s[0] + vector_1.s[0],
                            vector_0.s[1] + vector_1.s[1],
                            vector_0.s[2] + vector_1.s[2],
                            vector_0.s[3] + vector_1.s[3]};
    return result;
}
static inline float4 float4_mul(const float4 vector_0, const float4 vector_1) {
    const float4 result = { vector_0.s[0] * vector_1.s[0],
                            vector_0.s[1] * vector_1.s[1],
                            vector_0.s[2] * vector_1.s[2],
                            vector_0.s[3] * vector_1.s[3]};
    return result;
}
#endif //OPENCL_FLOAT4_H
