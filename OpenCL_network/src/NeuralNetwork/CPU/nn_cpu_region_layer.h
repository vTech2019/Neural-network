//
// Created by human on 19.03.2020.
//

#ifndef OPENCL_NN_CPU_REGION_LAYER_H
#define OPENCL_NN_CPU_REGION_LAYER_H

#include "nn_cpu_network.h"
#include "../Parser/cfg_net.h"
struct nn_cpu_region_layer;
typedef struct nn_cpu_region_layer nn_cpu_region_layer;
struct nn_cpu_region_layer{
    nn_cpu_layer* next_layer;
    NN_LAYER_TYPE type_layer;

    void (*forward)(nn_cpu_region_layer*, nn_cpu_network*);
    void (*backward)(nn_cpu_region_layer*, nn_cpu_network*);
    void (*update)(nn_cpu_region_layer*);
    float* output;

    unsigned width;
    unsigned height;
    unsigned channels;
    unsigned batch;

    unsigned output_height;
    unsigned output_width;
    unsigned output_channels;
    unsigned output_image_length;
    unsigned input_image_length;
    unsigned number_biases;
    unsigned num;
    unsigned coords;
    unsigned classes;
    unsigned background;
    unsigned softmax;
    unsigned truths;
    unsigned noobject_scale;
    unsigned thresh;
    unsigned bias_match;
    unsigned object_scale;
    unsigned rescore;
    unsigned coord_scale;
    unsigned class_scale;
    unsigned mask_scale;

    float* softmax_tree; ////////////////////
   // unsigned softmax_tree;

    float cost;

    float* bias;
    float* bias_update;
    float* delta;

    float biases[0];
};
void nn_cpu_init_region_network(cfg_darknet_region_layer_info* config, nn_cpu_region_layer* layer);
#endif //OPENCL_NN_CPU_REGION_LAYER_H
