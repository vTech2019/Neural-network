//
// Created by human on 12.03.2020.
//

#include "nn_cpu_activation.h"

float nn_cpu_get_activation_function(NN_ACTIVATION_FUNCTION type_function, float x)
{
    switch(type_function){
        case NN_ELU:        return nn_cpu_activation_elu(x);
        case NN_HARDTAN:    return nn_cpu_activation_hardtan(x);
        case NN_LEAKY:      return nn_cpu_activation_leaky(x);
        case NN_LHTAN:      return nn_cpu_activation_lhtan(x);
        case NN_LINEAR:     return nn_cpu_activation_linear(x);
        case NN_LOGGY:      return nn_cpu_activation_loggy(x);
        case NN_LOGISTIC:   return nn_cpu_activation_logistic(x);
        case NN_PLSE:       return nn_cpu_activation_plse(x);
        case NN_RAMP:       return nn_cpu_activation_ramp(x);
        case NN_RELIE:      return nn_cpu_activation_relie(x);
        case NN_RELU:       return nn_cpu_activation_relu(x);
        case NN_SELU:       return nn_cpu_activation_selu(x);
        case NN_STAIR:      return nn_cpu_activation_stair(x);
        case NN_TANH:       return nn_cpu_activation_tanh(x);
        default:            return 0.0f;
    }
}
float nn_cpu_get_gradient_function(NN_ACTIVATION_FUNCTION type_function, float x)
{
    switch(type_function){
        case NN_ELU:        return nn_cpu_gradient_elu(x);
        case NN_HARDTAN:    return nn_cpu_gradient_hardtan(x);
        case NN_LEAKY:      return nn_cpu_gradient_leaky(x);
        case NN_LHTAN:      return nn_cpu_gradient_lhtan(x);
        case NN_LINEAR:     return nn_cpu_gradient_linear(x);
        case NN_LOGGY:      return nn_cpu_gradient_loggy(x);
        case NN_LOGISTIC:   return nn_cpu_gradient_logistic(x);
        case NN_PLSE:       return nn_cpu_gradient_plse(x);
        case NN_RAMP:       return nn_cpu_gradient_ramp(x);
        case NN_RELIE:      return nn_cpu_gradient_relie(x);
        case NN_RELU:       return nn_cpu_gradient_relu(x);
        case NN_SELU:       return nn_cpu_gradient_selu(x);
        case NN_STAIR:      return nn_cpu_gradient_stair(x);
        case NN_TANH:       return nn_cpu_gradient_tanh(x);
        default:            return 0.0f;
    }
}
void nn_cpu_activate(float* data, size_t length, NN_ACTIVATION_FUNCTION type_function){
    size_t i = 0;
    for (; i + 7 < length; i += 8){
        data[i] = nn_cpu_get_activation_function(type_function, data[i]);
        data[i + 1] = nn_cpu_get_activation_function(type_function, data[i + 1]);
        data[i + 2] = nn_cpu_get_activation_function(type_function, data[i + 2]);
        data[i + 3] = nn_cpu_get_activation_function(type_function, data[i + 3]);
        data[i + 4] = nn_cpu_get_activation_function(type_function, data[i + 4]);
        data[i + 5] = nn_cpu_get_activation_function(type_function, data[i + 5]);
        data[i + 6] = nn_cpu_get_activation_function(type_function, data[i + 6]);
        data[i + 7] = nn_cpu_get_activation_function(type_function, data[i + 7]);
    }
    for (; i < length; i++)
        data[i] = nn_cpu_get_activation_function(type_function, data[i]);
}
void nn_cpu_gradient(float* data, float* delta, size_t length, NN_ACTIVATION_FUNCTION type_function){
    size_t i = 0;
    for (; i + 7 < length; i += 8){
        delta[i] *= nn_cpu_get_gradient_function(type_function, data[i]);
        delta[i + 1] *= nn_cpu_get_gradient_function(type_function, data[i + 1]);
        delta[i + 2] *= nn_cpu_get_gradient_function(type_function, data[i + 2]);
        delta[i + 3] *= nn_cpu_get_gradient_function(type_function, data[i + 3]);
        delta[i + 4] *= nn_cpu_get_gradient_function(type_function, data[i + 4]);
        delta[i + 5] *= nn_cpu_get_gradient_function(type_function, data[i + 5]);
        delta[i + 6] *= nn_cpu_get_gradient_function(type_function, data[i + 6]);
        delta[i + 7] *= nn_cpu_get_gradient_function(type_function, data[i + 7]);
    }
    for (; i < length; i++)
        delta[i] *= nn_cpu_get_gradient_function(type_function, data[i]);
}