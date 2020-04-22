//
// Created by human on 18.03.2020.
//

#include "nn_cpu_connected_layer.h"

void nn_cpu_init_connected_network(cfg_darknet_connected_layer_info* config, nn_cpu_connected_layer* layer){
    layer->input_channels = config->input_channels;
    layer->height = config->height;
    layer->width = config->width;
    layer->batch = config->batch;
    layer->output_channels = config->output_channels;
    layer->output_height = config->output_height;
    layer->output_width = config->output_width;
    layer->output_length = config->output_length;
    layer->input_length = config->input_length;
    layer->stride = config->stride;
    layer->batch_normalize = config->batch_normalize;
    layer->workspace_size = config->workspace_size;

    layer->function = config->function;
}