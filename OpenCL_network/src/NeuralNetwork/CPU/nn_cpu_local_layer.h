//
// Created by human on 19.03.2020.
//

#ifndef OPENCL_NN_CPU_LOCAL_LAYER_H
#define OPENCL_NN_CPU_LOCAL_LAYER_H

#include "nn_cpu_network.h"
struct nn_cpu_local_layer;
typedef struct nn_cpu_local_layer nn_cpu_local_layer;
struct nn_cpu_local_layer{
    nn_cpu_layer* next_layer;
    NN_LAYER_TYPE type_layer;
    void (*forward)(nn_cpu_local_layer*, nn_cpu_network*);
    void (*backward)(nn_cpu_local_layer*, nn_cpu_network*);
    void (*update)(nn_cpu_local_layer*);
    float* output;

    unsigned workspace_size;

    unsigned input_width;
    unsigned input_height;
    unsigned input_channels;
    unsigned batch;

    unsigned filters;
    unsigned filter_length;
    unsigned stride;
    unsigned pad;

    unsigned output_height;
    unsigned output_width;
    unsigned output_channels;

    unsigned input_image_length;
    unsigned output_image_length;

    float* weight;
    float* weight_update;
    float* bias;
    float* bias_update;
    float* delta;

    NN_ACTIVATION_FUNCTION function;
};
void nn_cpu_init_local_network(cfg_darknet_local_layer_info* config, nn_cpu_local_layer* layer);
#endif //OPENCL_NN_CPU_LOCAL_LAYER_H
