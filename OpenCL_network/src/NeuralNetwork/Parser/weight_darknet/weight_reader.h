//
// Created by human on 15.03.2020.
//

#ifndef OPENCL_WEIGHT_READER_H
#define OPENCL_WEIGHT_READER_H

#include "../../nn_layers.h"
#include "../cfg_net.h"

struct cfg_darknet_weight_layer_info;
typedef struct cfg_darknet_weight_layer_info cfg_darknet_weight_layer_info;

struct cfg_darknet_weight_layer_info{
    cfg_darknet_weight_layer_info *next_layer;
    NN_LAYER_TYPE type_layer;
    size_t index_layer;
    void *data;
};

typedef struct{
    cfg_darknet_weight_layer_info *next_layer;
    NN_LAYER_TYPE type_layer;
    size_t index_layer;
    float* weights;
    float* biases;
    float* rolling_variance;
    float* rolling_mean;
    float* scales;
    size_t length_weight;
    size_t length_biases;
    size_t length_rolling_variance;
    size_t length_rolling_mean;
    size_t length_scales;
}weight_darknet_network_convolution_info;

typedef struct{
    cfg_darknet_weight_layer_info *next_layer;
    NN_LAYER_TYPE type_layer;
    size_t index_layer;
    float* weights;
    float* biases;
    size_t length_weight;
    size_t length_biases;
}weight_darknet_network_local_info;

typedef struct{
    cfg_darknet_weight_layer_info *next_layer;
    NN_LAYER_TYPE type_layer;
    size_t index_layer;
    float* scales;
    float* rolling_mean;
    float* rolling_variance;
    size_t length_scales;
    size_t length_rolling_mean;
    size_t length_rolling_variance;
}weight_darknet_network_batchnorm_info;

typedef struct{
    cfg_darknet_weight_layer_info *next_layer;
    NN_LAYER_TYPE type_layer;
    size_t index_layer;
    float* biases;
    float* weights;
    float* scales;
    float* rolling_mean;
    float* rolling_variance;
    size_t length_weight;
    size_t length_biases;
    size_t length_scales;
    size_t length_rolling_mean;
    size_t length_rolling_variance;
}weight_darknet_network_connected_info;

typedef struct weight_reader{
    size_t seen;
    size_t size_data_malloc;
    cfg_darknet_weight_layer_info* current_layer;
}weight_darknet_network_info;

weight_darknet_network_info* weights_reader_malloc(cfg_darknet_network_info* cfg, char *filename);
void weights_reader_free(weight_darknet_network_info* weight);

#endif //OPENCL_WEIGHT_READER_H
