//
// Created by human on 31.03.2020.
//

#ifndef OPENCL_INT4_H
#define OPENCL_INT4_H

#include <stdalign.h>

struct int4;
typedef struct int4 int4;
struct int4{
    alignas(16) int s[4];
};
#endif //OPENCL_INT4_H
