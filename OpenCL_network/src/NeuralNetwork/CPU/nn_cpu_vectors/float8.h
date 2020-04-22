//
// Created by human on 31.03.2020.
//

#ifndef OPENCL_FLOAT8_H
#define OPENCL_FLOAT8_H
#include <stdalign.h>

struct float8;
typedef struct float8 float8;
struct float8{
    alignas(32) float s[8];
};

static inline float8 float8_add(const float8 vector_0, const float8 vector_1) {
    const float8 result = { vector_0.s[0] + vector_1.s[0],
                            vector_0.s[1] + vector_1.s[1],
                            vector_0.s[2] + vector_1.s[2],
                            vector_0.s[3] + vector_1.s[3],
                            vector_0.s[4] + vector_1.s[4],
                            vector_0.s[5] + vector_1.s[5],
                            vector_0.s[6] + vector_1.s[6],
                            vector_0.s[7] + vector_1.s[7]};
    return result;
}
static inline float8 float8_mul(const float8 vector_0, const float8 vector_1) {
    const float8 result = { vector_0.s[0] * vector_1.s[0],
                            vector_0.s[1] * vector_1.s[1],
                            vector_0.s[2] * vector_1.s[2],
                            vector_0.s[3] * vector_1.s[3],
                            vector_0.s[4] * vector_1.s[4],
                            vector_0.s[5] * vector_1.s[5],
                            vector_0.s[6] * vector_1.s[6],
                            vector_0.s[7] * vector_1.s[7]};
    return result;
}
static inline float8 float8_broadcast_value(const float value) {
    const float8 result = { value, value,
                            value, value,
                            value, value,
                            value, value};
    return result;
}
#endif //OPENCL_FLOAT8_H
