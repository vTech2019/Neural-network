//
// Created by human on 19.03.2020.
//

#include "nn_cpu_local_layer.h"

void nn_cpu_init_local_network(cfg_darknet_local_layer_info* config, nn_cpu_local_layer* layer){
    layer->workspace_size = config->workspace_size;
    layer->input_width = config->input_width;
    layer->input_height = config->input_height;
    layer->input_channels = config->input_channels;
    layer->batch = config->batch;
    layer->filters = config->filters;
    layer->filter_length = config->filter_size;
    layer->stride = config->stride;
    layer->pad = config->pad;
    layer->output_height = config->output_height;
    layer->output_width = config->output_width;
    layer->output_channels = config->output_channels;
    layer->input_image_length = config->input_image_length;
    layer->output_image_length = config->output_image_length;
    layer->function = config->function;
}