//
// Created by human on 12.03.2020.
//

#ifndef OPENCL_NN_ACTIVATION_H
#define OPENCL_NN_ACTIVATION_H

#include <math.h>
#include <stdio.h>
#include "../nn_layers.h"

static inline float nn_cpu_activation_linear(float x){return x;}
static inline float nn_cpu_activation_logistic(float x){return 1.0f/(1.0f + expf(-x));}
static inline float nn_cpu_activation_loggy(float x){return 2.0f/(1.0f + expf(-x)) - 1.0f;}
static inline float nn_cpu_activation_relu(float x){return x * (x > 0.0f ? 1.0f : 0.0f) ;}
static inline float nn_cpu_activation_elu(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
static inline float nn_cpu_activation_selu(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);}
static inline float nn_cpu_activation_relie(float x){return (x>0) ? x : .01*x;}
static inline float nn_cpu_activation_ramp(float x){return x*(x>0)+.1*x;}
static inline float nn_cpu_activation_leaky(float x){return (x>0) ? x : .1*x;}
static inline float nn_cpu_activation_tanh(float x){return (exp(2*x)-1)/(exp(2*x)+1);}
static inline float nn_cpu_activation_plse(float x)
{
    if(x < -4) return 0.01f * (x + 4);
    if(x > 4)  return 0.01f * (x - 4) + 1;
    return 0.125f*x + 0.5f;
}
static inline float nn_cpu_activation_stair(float x)
{
    int n = floorf(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}
static inline float nn_cpu_activation_hardtan(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float nn_cpu_activation_lhtan(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}


static float nn_cpu_gradient_lhtan(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

static float nn_cpu_gradient_hardtan(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
static float nn_cpu_gradient_linear(){return 1;}
static float nn_cpu_gradient_logistic(float x){return (1-x)*x;}
static float nn_cpu_gradient_loggy(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}
static float nn_cpu_gradient_stair(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}
static float nn_cpu_gradient_relu(float x){return (x>0.0f);}
static float nn_cpu_gradient_elu(float x){return (x >= 0.0f) + (x < 0.0f)*(x + 1);}
static float nn_cpu_gradient_selu(float x){return (x >= 0.0f)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
static float nn_cpu_gradient_relie(float x){return (x>0) ? 1 : .01;}
static float nn_cpu_gradient_ramp(float x){return (x>0.0f)+.1;}
static float nn_cpu_gradient_leaky(float x){return (x>0) ? 1.0f : 0.1f;}
static float nn_cpu_gradient_tanh(float x){return 1-x*x;}
static float nn_cpu_gradient_plse(float x){return (x < 0 || x > 1) ? 0.01f : 0.125f;}

void nn_cpu_activate(float* data, size_t length, NN_ACTIVATION_FUNCTION type_function);
void nn_cpu_gradient(float* data, float* delta, size_t length, NN_ACTIVATION_FUNCTION type_function);

float nn_cpu_get_activation_function(NN_ACTIVATION_FUNCTION type_function, float x);
float nn_cpu_get_gradient_function(NN_ACTIVATION_FUNCTION type_function, float x);
#endif //OPENCL_NN_ACTIVATION_H
