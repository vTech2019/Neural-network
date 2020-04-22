//
// Created by human on 19.03.2020.
//

#ifndef OPENCL_NN_CPU_MAXPOOL_LAYER_H
#define OPENCL_NN_CPU_MAXPOOL_LAYER_H

#include "nn_cpu_network.h"
struct nn_cpu_maxpool_layer;
typedef struct nn_cpu_maxpool_layer nn_cpu_maxpool_layer;
struct nn_cpu_maxpool_layer{
    nn_cpu_layer* next_layer;
    NN_LAYER_TYPE type_layer;
    void (*forward)(nn_cpu_maxpool_layer*, nn_cpu_network*);
    void (*backward)(nn_cpu_maxpool_layer*, nn_cpu_network*);
    void (*update)(nn_cpu_maxpool_layer*);
    float* output;

    unsigned width;
    unsigned height;
    unsigned channels;
    unsigned batch;

    unsigned stride;
    unsigned filter_length;
    unsigned input_image_length;
    unsigned padding;
    unsigned output_width;
    unsigned output_height;
    unsigned output_channels;
    unsigned output_image_length;
    unsigned output_batch_length;

    unsigned* indices;
    float* delta;
};
void nn_cpu_init_maxpool_network(cfg_darknet_maxpool_layer_info* config, nn_cpu_maxpool_layer* layer);
#endif //OPENCL_NN_CPU_MAXPOOL_LAYER_H
