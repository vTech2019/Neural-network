//
// Created by human on 18.03.2020.
//

#ifndef OPENCL_NN_CPU_CONNECTED_LAYER_H
#define OPENCL_NN_CPU_CONNECTED_LAYER_H

#include "nn_cpu_network.h"
struct nn_cpu_connected_layer;
typedef struct nn_cpu_connected_layer nn_cpu_connected_layer;
struct nn_cpu_connected_layer{
    nn_cpu_layer* next_layer;
    NN_LAYER_TYPE type_layer;
    void (*forward)(nn_cpu_connected_layer*, nn_cpu_network*);
    void (*backward)(nn_cpu_connected_layer*, nn_cpu_network*);
    void (*update)(nn_cpu_connected_layer*);
    float* output;

    unsigned width;
    unsigned height;
    unsigned input_channels;
    unsigned batch;

    unsigned output_width;
    unsigned output_height;
    unsigned output_channels;

    unsigned output_length;
    unsigned input_length;

    unsigned batch_normalize;

    unsigned stride;

    unsigned workspace_size;

    float* bias;
    float* bias_update;
    float* weight;
    float* weight_update;
    float* delta;
    float* input;

    float* scales;
    float* scales_update;
    float* mean;
    float* variance; //dispersion
    float* mean_delta;
    float* variance_delta;
    float* rolling_mean;
    float* rolling_variance;
    float* x;
    float* x_normalize;

    NN_ACTIVATION_FUNCTION function;
};
void nn_cpu_init_connected_network(cfg_darknet_connected_layer_info* config, nn_cpu_connected_layer* layer);

#endif //OPENCL_NN_CPU_CONNECTED_LAYER_H
