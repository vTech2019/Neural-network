//
// Created by human on 11.03.2020.
//

#ifndef OPENCL_NN_CONVOLUTION_LAYER_H
#define OPENCL_NN_CONVOLUTION_LAYER_H
#include "../nn_layers.h"
#include "../Parser/cfg_net.h"
#include "nn_cpu_network.h"
struct nn_cpu_convolutional_layer;
typedef struct nn_cpu_convolutional_layer nn_cpu_convolutional_layer;
struct nn_cpu_convolutional_layer{
    nn_cpu_layer* next_layer;
    NN_LAYER_TYPE type_layer;
    void (*forward)(nn_cpu_convolutional_layer*, nn_cpu_network*);
    void (*backward)(nn_cpu_convolutional_layer*, nn_cpu_network*);
    void (*update)(nn_cpu_convolutional_layer*);
    float* output;

    unsigned batch;

    unsigned input_length;
    unsigned width;
    unsigned height;
    unsigned channels;

    unsigned output_length;
    unsigned output_width;
    unsigned output_height;
    unsigned output_channels;

    unsigned filters;
    unsigned filter_length;
    unsigned filters_weight_length;
    unsigned stride;
    unsigned padding;
    unsigned groups;
    unsigned batch_normalize;
    unsigned transposed;

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
void nn_cpu_init_convolution_network(cfg_darknet_convolution_layer_info* config, nn_cpu_convolutional_layer* layer);

//void nn_convolutional_layer_create(nn_convolutional_layer* layer, int batch, int height, int width, int channels, int filters, int groups, int filter_size, int stride, int padding, float (*activation)(float x), int batch_normalize, int binary, int xnor, int adam);


#endif //OPENCL_NN_CONVOLUTION_LAYER_H
