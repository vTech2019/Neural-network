//
// Created by human on 30.03.2020.
//

#ifndef OPENCL_LONGLONG2_H
#define OPENCL_LONGLONG2_H
#include <stdalign.h>
struct longlong2;
typedef struct longlong2 longlong2;

struct longlong2{
    alignas(16) long long s[2];
};
#endif //OPENCL_LONGLONG2_H
