//
// Created by human on 13.03.2020.
//

#ifndef OPENCL_NN_CPU_NETWORK_H
#define OPENCL_NN_CPU_NETWORK_H
#include "../Parser/cfg_net.h"
#include "../nn_network.h"
typedef struct nn_cpu_layer nn_cpu_layer;
struct nn_cpu_layer{
    nn_cpu_layer* next_layer;
    NN_LAYER_TYPE type_layer;
    void (*forward)(void*, void*);
    void (*backward)(void*, void*);
    void (*update)(void*);
    float* output;
    void* data;
};


typedef struct nn_cpu_network{
    size_t size_network;
    unsigned number_layers;

    unsigned width;
    unsigned height;
    unsigned channels;
    unsigned batch;
    unsigned train;

    unsigned seen;
    unsigned max_workspace_size;
    float learning_rate;
    float cost;
    float* truth;
    float* delta;
    float* input;
    float* workspace;
    nn_cpu_layer* layers;
} nn_cpu_network;

nn_cpu_network* nn_cpu_network_malloc(nn_network_config* config_network, nn_network_weight* weight_network);
void nn_cpu_network_free(nn_cpu_network* network);
void nn_cpu_forward_network(nn_cpu_network *network);
#endif //OPENCL_NN_CPU_NETWORK_H
