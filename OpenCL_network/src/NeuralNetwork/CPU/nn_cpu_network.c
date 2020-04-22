//
// Created by human on 13.03.2020.
//
#include <malloc.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "nn_cpu_network.h"
#include "nn_cpu_convolution_layer.h"
#include "nn_cpu_connected_layer.h"
#include "nn_cpu_local_layer.h"
#include "nn_cpu_maxpool_layer.h"
#include "nn_cpu_region_layer.h"
#include "nn_cpu_operations.h"
#define ALIGN 64
size_t get_size_network(nn_network_config* config_network){
    size_t size = 0;
    size += ALIGN;
    size += config_network->batch * config_network->channels * config_network->height * config_network->width * sizeof(float); //input
    size += ALIGN;
    size += config_network->max_workspace_size;
    cfg_darknet_layer_info* start_layer = config_network->layers;
    for (size_t i = 0; i < config_network->number_layers; i++) {
        switch (start_layer->type_layer){
            case CFG_PARSER_TYPE_DECONVOLUTION:
                break;
            case CFG_PARSER_TYPE_CONVOLUTION:
            {
                cfg_darknet_convolution_layer_info* config = (cfg_darknet_convolution_layer_info*)start_layer;
                size += ALIGN + config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //output
                size += config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //delta
                size += config->weight_length * sizeof(float);                                                                  //weight (k)
                size += config->weight_length * sizeof(float);                                                                  //weight_update (k)
                size += config->filters * sizeof(float);                                                                        //bias (b)
                size += config->filters * sizeof(float);                                                                        //bias_update (b)
                size += sizeof(nn_cpu_convolutional_layer);
                if (config->batch_normalize){
                    size += config->filters * sizeof(float); //scales
                    size += config->filters * sizeof(float); //scales_update
                    size += config->filters * sizeof(float); //mean
                    size += config->filters * sizeof(float); //variance (dispersion)
                    size += config->filters * sizeof(float); //mean_delta
                    size += config->filters * sizeof(float); //variance_delta (dispersion)
                    size += config->filters * sizeof(float); //rolling_mean
                    size += config->filters * sizeof(float); //rolling_variance
                    size += config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //x (not normalize)
                    size += config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //x_normalize
                }
                break;
            }
            case CFG_PARSER_TYPE_CONNECTED:{
                cfg_darknet_connected_layer_info* config = (cfg_darknet_connected_layer_info*)start_layer;
                size += config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //output
                size += config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //delta
                size += config->output_length * config->input_length * sizeof(float);                                                       //weight (k)
                size += config->output_length * config->input_length * sizeof(float);                                                       //weight_update (k)
                size += config->output_length * sizeof(float);                                                                        //bias (b)
                size += config->output_length * sizeof(float);                                                                        //bias_update (b)
                size += sizeof(nn_cpu_connected_layer);
                if (config->batch_normalize){
                    size += config->output_length * sizeof(float); //scales
                    size += config->output_length * sizeof(float); //scales_update
                    size += config->output_length * sizeof(float); //mean
                    size += config->output_length * sizeof(float); //variance (dispersion)
                    size += config->output_length * sizeof(float); //mean_delta
                    size += config->output_length * sizeof(float); //variance_delta (dispersion)
                    size += config->output_length * sizeof(float); //rolling_mean
                    size += config->output_length * sizeof(float); //rolling_variance
                    size += config->batch * config->output_length * sizeof(float); //x (not normalize)
                    size += config->batch * config->output_length * sizeof(float); //x_normalize
                }
                break;
            }
            case CFG_PARSER_TYPE_LOCAL:{
                cfg_darknet_local_layer_info* config = (cfg_darknet_local_layer_info*)start_layer;
                size += config->output_channels * config->output_height * config->output_width * config->filters * config->filter_size * config->filter_size * sizeof(float); //weight (k)
                size += config->output_channels * config->output_height * config->output_width * config->filters * config->filter_size * config->filter_size * sizeof(float); //weight_update (k)
                size += config->output_width * config->output_height * config->output_channels * sizeof(float);                                                               //bias (b)
                size += config->output_width * config->output_height * config->output_channels * sizeof(float);                                                               //bias_update (b)
                size += config->batch * config->filters * config->output_height * config->output_width * sizeof(float);                                                       //output
                size += config->batch * config->filters * config->output_height * config->output_width * sizeof(float);                                                       //delta                                                                    //bias_update (b)
                size += sizeof(nn_cpu_local_layer);
                break;
            }
            case CFG_PARSER_TYPE_MAXPOOL: {
                cfg_darknet_maxpool_layer_info* config = (cfg_darknet_maxpool_layer_info*)start_layer;
                size += config->batch * config->output_width * config->output_height * config->output_channels * sizeof(float); //indices
                size += config->batch * config->output_width * config->output_height * config->output_channels * sizeof(float); //output
                size += config->batch * config->output_width * config->output_height * config->output_channels * sizeof(float); //delta
                size += sizeof(nn_cpu_maxpool_layer);
                break;
            }
            case CFG_PARSER_TYPE_REGION: {
                cfg_darknet_region_layer_info* config = (cfg_darknet_region_layer_info*)start_layer;
                //size += sizeof(float); //cost
                size += config->num * 2 * sizeof(float); //bias
                size += config->num * 2 * sizeof(float); //bias_update
                size += config->num * config->height * config->width * config->num * (config->classes + config->coords + 1) * sizeof(float); //delta
                size += config->num * config->height * config->width * config->num * (config->classes + config->coords + 1) * sizeof(float); //output

                size += sizeof(nn_cpu_region_layer);
                break;
            }
                //case CFG_PARSER_TYPE_BATCHNORM:
                //    break;
            default: break;
        }
        start_layer = start_layer->next_layer;
    }
    return size;
}
size_t init_pointers(nn_network_config* config_network, nn_cpu_network* network, size_t size_parameters_network) {
    char* init_ptr = ((char*)network) + size_parameters_network;
    init_ptr = (char *) ((size_t) (init_ptr + (ALIGN - 1)) & -ALIGN);
    network->input = (float *) init_ptr;
    init_ptr += config_network->batch * config_network->channels * config_network->height * config_network->width * sizeof(float); //input
    init_ptr = (char *) ((size_t) (init_ptr + (ALIGN - 1)) & -ALIGN);
    network->workspace = init_ptr;
    init_ptr += config_network->max_workspace_size;
    network->layers = (nn_cpu_layer *) init_ptr;
    network->number_layers = config_network->number_layers;
    nn_cpu_layer* start_layer = (nn_cpu_layer *) init_ptr;
    cfg_darknet_layer_info* start_config_layer = config_network->layers;
    for (size_t i = 0; i < config_network->number_layers; i++) {
        switch (start_config_layer->type_layer){
            case CFG_PARSER_TYPE_DECONVOLUTION:
                break;
            case CFG_PARSER_TYPE_CONVOLUTION:
            {
                cfg_darknet_convolution_layer_info* config = (cfg_darknet_convolution_layer_info*)start_config_layer;
                nn_cpu_convolutional_layer* layer = start_layer;
                layer->type_layer = NN_CONVOLUTIONAL;
                init_ptr += sizeof(nn_cpu_convolutional_layer);
                init_ptr = (char *) ((size_t) (init_ptr + (ALIGN - 1)) & -ALIGN);
                layer->output = (float *) init_ptr;
                init_ptr += config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //output
                printf("%zu = %zx   %zx\n", i , layer->output, init_ptr);
                layer->delta = (float *) init_ptr;
                init_ptr += config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //delta
                layer->weight = (float *) init_ptr;
                init_ptr += config->weight_length * sizeof(float);                                                                  //weight (k)
                layer->weight_update = (float *) init_ptr;
                init_ptr += config->weight_length * sizeof(float);                                                                  //weight_update (k)
                layer->bias = (float *) init_ptr;
                init_ptr += config->filters * sizeof(float);                                                                        //bias (b)
                layer->bias_update = (float *) init_ptr;
                init_ptr += config->filters * sizeof(float);                                                                        //bias_update (b)
                if (config->batch_normalize){
                    layer->scales = (float *) init_ptr;
                    init_ptr += config->filters * sizeof(float); //scales
                    layer->scales_update = (float *) init_ptr;
                    init_ptr += config->filters * sizeof(float); //scales_update
                    layer->mean = (float *) init_ptr;
                    init_ptr += config->filters * sizeof(float); //mean
                    layer->variance = (float *) init_ptr;
                    init_ptr += config->filters * sizeof(float); //variance (dispersion)
                    layer->mean_delta = (float *) init_ptr;
                    init_ptr += config->filters * sizeof(float); //mean_delta
                    layer->variance_delta = (float *) init_ptr;
                    init_ptr += config->filters * sizeof(float); //variance_delta (dispersion)
                    layer->rolling_mean = (float *) init_ptr;
                    init_ptr += config->filters * sizeof(float); //rolling_mean
                    layer->rolling_variance = (float *) init_ptr;
                    init_ptr += config->filters * sizeof(float); //rolling_variance
                    layer->x = (float *) init_ptr;
                    init_ptr += config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //x (not normalize)
                    layer->x_normalize = (float *) init_ptr;
                    init_ptr += config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //x_normalize
                }
                layer->next_layer = (nn_cpu_layer *) init_ptr;
                start_layer = (nn_cpu_layer *) init_ptr;
                break;
            }
            case CFG_PARSER_TYPE_CONNECTED:{
                cfg_darknet_connected_layer_info* config = (cfg_darknet_connected_layer_info*)start_config_layer;
                nn_cpu_connected_layer* layer = start_layer;
                layer->type_layer = NN_CONNECTED;
                init_ptr += sizeof(nn_cpu_connected_layer);
                layer->output = (float *) init_ptr;
                init_ptr += config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //output
                layer->delta = (float *) init_ptr;
                init_ptr += config->batch * config->output_channels * config->output_height * config->output_width * sizeof(float); //delta
                layer->weight = (float *) init_ptr;
                init_ptr += config->output_length * config->input_length * sizeof(float);                                                       //weight (k)
                layer->weight_update = (float *) init_ptr;
                init_ptr += config->output_length * config->input_length * sizeof(float);                                                       //weight_update (k)
                layer->bias = (float *) init_ptr;
                init_ptr += config->output_length * sizeof(float);                                                                        //bias (b)
                layer->bias_update = (float *) init_ptr;
                init_ptr += config->output_length * sizeof(float);                                                                        //bias_update (b)
                if (config->batch_normalize){
                    layer->scales = (float *) init_ptr;
                    init_ptr += config->output_length * sizeof(float); //scales
                    layer->scales_update = (float *) init_ptr;
                    init_ptr += config->output_length * sizeof(float); //scales_update
                    layer->mean = (float *) init_ptr;
                    init_ptr += config->output_length * sizeof(float); //mean
                    layer->variance = (float *) init_ptr;
                    init_ptr += config->output_length * sizeof(float); //variance (dispersion)
                    layer->mean_delta = (float *) init_ptr;
                    init_ptr += config->output_length * sizeof(float); //mean_delta
                    layer->variance_delta = (float *) init_ptr;
                    init_ptr += config->output_length * sizeof(float); //variance_delta (dispersion)
                    layer->rolling_mean = (float *) init_ptr;
                    init_ptr += config->output_length * sizeof(float); //rolling_mean
                    layer->rolling_variance = (float *) init_ptr;
                    init_ptr += config->output_length * sizeof(float); //rolling_variance
                    layer->x = (float *) init_ptr;
                    init_ptr += config->batch * config->output_length * sizeof(float); //x (not normalize)
                    layer->x_normalize = (float *) init_ptr;
                    init_ptr += config->batch * config->output_length * sizeof(float); //x_normalize
                }
                layer->next_layer = (nn_cpu_layer *) init_ptr;
                start_layer = (nn_cpu_layer *) init_ptr;
                break;
            }
            case CFG_PARSER_TYPE_LOCAL:{
                cfg_darknet_local_layer_info* config = (cfg_darknet_local_layer_info*)start_config_layer;
                nn_cpu_local_layer* layer = start_layer;
                layer->type_layer = NN_LOCAL;
                init_ptr += sizeof(nn_cpu_local_layer);
                layer->weight = (float *) init_ptr;
                init_ptr += config->output_channels * config->output_height * config->output_width * config->filters * config->filter_size * config->filter_size * sizeof(float); //weight (k)
                layer->weight_update = (float *) init_ptr;
                init_ptr += config->output_channels * config->output_height * config->output_width * config->filters * config->filter_size * config->filter_size * sizeof(float); //weight_update (k)
                layer->bias = (float *) init_ptr;
                init_ptr += config->output_width * config->output_height * config->output_channels * sizeof(float);                                                               //bias (b)
                layer->bias_update = (float *) init_ptr;
                init_ptr += config->output_width * config->output_height * config->output_channels * sizeof(float);                                                               //bias_update (b)
                layer->output = (float *) init_ptr;
                init_ptr += config->batch * config->filters * config->output_height * config->output_width * sizeof(float);                                                       //output
                layer->delta = (float *) init_ptr;
                init_ptr += config->batch * config->filters * config->output_height * config->output_width * sizeof(float);                                                       //delta
                layer->next_layer = (nn_cpu_layer *) init_ptr;
                start_layer = (nn_cpu_layer *) init_ptr;

                break;
            }
            case CFG_PARSER_TYPE_MAXPOOL: {
                cfg_darknet_maxpool_layer_info* config = (cfg_darknet_maxpool_layer_info*)start_config_layer;
                nn_cpu_maxpool_layer* layer = start_layer;
                layer->type_layer = NN_MAXPOOL;
                init_ptr += sizeof(nn_cpu_maxpool_layer);
                layer->indices = (float *) init_ptr;
                init_ptr +=  config->batch * config->output_width * config->output_height * config->output_channels * sizeof(float); //indices
                layer->output = (float *) init_ptr;
                init_ptr +=  config->batch * config->output_width * config->output_height * config->output_channels * sizeof(float); //output
                layer->delta = (float *) init_ptr;
                init_ptr +=  config->batch * config->output_width * config->output_height * config->output_channels * sizeof(float); //delta
                layer->next_layer = (nn_cpu_layer *) init_ptr;
                start_layer = (nn_cpu_layer *) init_ptr;
                break;
            }
            case CFG_PARSER_TYPE_REGION: {
                cfg_darknet_region_layer_info* config = (cfg_darknet_region_layer_info*)start_config_layer;
                nn_cpu_region_layer* layer = start_layer;
                layer->type_layer = NN_REGION;
                init_ptr += sizeof(nn_cpu_region_layer);
                init_ptr += config->number_biases * sizeof(float);
                //init_ptr += sizeof(float); //cost
                layer->bias = init_ptr;
                init_ptr += config->num * 2 * sizeof(float); //bias
                layer->bias_update = init_ptr;
                init_ptr += config->num * 2 * sizeof(float); //bias_update
                layer->delta = init_ptr;
                init_ptr += config->num * config->height * config->width * config->num * (config->classes + config->coords + 1) * sizeof(float); //delta
                layer->output = init_ptr;
                init_ptr += config->num * config->height * config->width * config->num * (config->classes + config->coords + 1) * sizeof(float); //output
                layer->next_layer = (nn_cpu_layer *) init_ptr;
                start_layer = (nn_cpu_layer *) init_ptr;
                break;
            }
                //case CFG_PARSER_TYPE_BATCHNORM:
                //    break;
            default: break;
        }
        start_config_layer = start_config_layer->next_layer;
    }
}
void init_network(nn_network_config* config_network, nn_cpu_network* network) {
    network->batch = config_network->batch;
    network->height = config_network->height;
    network->width = config_network->width;
    network->channels = config_network->channels;
    network->max_workspace_size = config_network->max_workspace_size;
    network->number_layers = config_network->number_layers;
    network->learning_rate = config_network->learning_rate;

    cfg_darknet_layer_info* start_config_layer = config_network->layers;
    nn_cpu_layer* start_layer = network->layers;
    for (size_t i = 0; i < config_network->number_layers; i++) {
        switch (start_config_layer->type_layer){
            case CFG_PARSER_TYPE_DECONVOLUTION:
                break;
            case CFG_PARSER_TYPE_CONVOLUTION:
            {
                cfg_darknet_convolution_layer_info* config = (cfg_darknet_convolution_layer_info*)start_config_layer;
                nn_cpu_convolutional_layer* layer = start_layer;
                nn_cpu_init_convolution_network(config, layer);
                break;
            }
            case CFG_PARSER_TYPE_CONNECTED:{
                cfg_darknet_connected_layer_info* config = (cfg_darknet_connected_layer_info*)start_config_layer;
                nn_cpu_connected_layer* layer = start_layer;
                nn_cpu_init_connected_network(config, layer);
                break;
            }
            case CFG_PARSER_TYPE_LOCAL:{
                cfg_darknet_local_layer_info* config = (cfg_darknet_local_layer_info*)start_config_layer;
                nn_cpu_local_layer* layer = start_layer;
                nn_cpu_init_local_network(config, layer);
                break;
            }
            case CFG_PARSER_TYPE_MAXPOOL: {
                cfg_darknet_maxpool_layer_info* config = (cfg_darknet_maxpool_layer_info*)start_config_layer;
                nn_cpu_maxpool_layer* layer = start_layer;
                nn_cpu_init_maxpool_network(config, layer);
                break;
            }
            case CFG_PARSER_TYPE_REGION: {
                cfg_darknet_region_layer_info* config = (cfg_darknet_region_layer_info*)start_config_layer;
                nn_cpu_region_layer* layer = start_layer;
                nn_cpu_init_region_network(config, layer);
                break;
            }
                //case CFG_PARSER_TYPE_BATCHNORM:
                //    break;
            default: break;
        }
        start_config_layer = start_config_layer->next_layer;
        start_layer = start_layer->next_layer;
    }
    network->size_network =(size_t)start_layer - (size_t)network;
}
void init_network_layers(nn_cpu_network* network, nn_network_weight* weight_network) {
    ptrdiff_t i, thread_data = 0;
    network->seen = weight_network->seen;
    cfg_darknet_weight_layer_info weight_layer = { NULL, 0, network->number_layers, NULL };
    nn_cpu_layer* start_layer = network->layers;
    cfg_darknet_weight_layer_info* start_weight_layer = weight_network ? weight_network->current_layer : &weight_layer;
#pragma omp parallel for firstprivate(thread_data, start_layer, start_weight_layer) shared(network, weight_network) private(i)
    for (i = 0; i < network->number_layers; i++) {
        for (size_t j = thread_data; j < i; j++){
            start_layer = start_layer->next_layer;
            if (start_weight_layer && start_weight_layer->index_layer < i)
                start_weight_layer = start_weight_layer->next_layer;
            thread_data = j;
        }
        switch (start_layer->type_layer){
            case NN_DECONVOLUTIONAL:
                break;
            case NN_CONVOLUTIONAL:{
                nn_cpu_convolutional_layer* config = (nn_cpu_convolutional_layer*)start_layer;
                float scale = sqrtf(2.0f / (config->filter_length*config->filter_length*config->channels/config->groups));
                if (start_weight_layer->index_layer == i){
                    weight_darknet_network_convolution_info* weight_layer = start_weight_layer;
                    if (start_weight_layer->type_layer != NN_CONVOLUTIONAL) {fprintf(stderr,"ERROR CONVOLUTION WEIGHT INIT: %zu INDEX LAYER", i); goto exit_function; }
                    if (weight_layer->length_weight != config->filters_weight_length) {fprintf(stderr, "ERROR CONVOLUTION WEIGHT INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_weight, config->filters_weight_length); goto exit_function; }
                    if (weight_layer->length_biases != config->filters) {fprintf(stderr,"ERROR CONVOLUTION BIAS INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_biases, config->filters); goto exit_function; }
                    memcpy(config->weight, weight_layer->weights, weight_layer->length_weight * sizeof(float));
                    memcpy(config->bias, weight_layer->biases, weight_layer->length_biases * sizeof(float));
                    if (config->batch_normalize) {
                        if (weight_layer->length_rolling_mean != config->filters) {fprintf(stderr,"ERROR CONVOLUTION ROLLING MEAN INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_rolling_mean, config->filters); goto exit_function; }
                        if (weight_layer->length_rolling_variance != config->filters) {fprintf(stderr,"ERROR CONVOLUTION ROLLING VARIANCE INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_rolling_variance, config->filters); goto exit_function; }
                        if (weight_layer->length_scales != config->filters) {fprintf(stderr,"ERROR CONVOLUTION SCALES INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_scales, config->filters); goto exit_function; }

                        memcpy(config->rolling_variance, weight_layer->rolling_mean, weight_layer->length_rolling_mean * sizeof(float));
                        memcpy(config->rolling_mean, weight_layer->rolling_variance, weight_layer->length_rolling_variance * sizeof(float));
                        memcpy(config->scales, weight_layer->scales, weight_layer->length_scales * sizeof(float));
                    }
                    start_weight_layer = start_weight_layer->next_layer;
                }else{
                    nn_cpu_set_random_float_linear(config->filters_weight_length, config->weight, scale, 0.0f);
                    nn_cpu_set_float_value_linear(config->filters, config->bias, 0.0f);
                    if (config->batch_normalize) {
                        nn_cpu_set_float_value_linear(config->filters, config->rolling_mean, 0.0f);
                        nn_cpu_set_float_value_linear(config->filters, config->rolling_variance, 0.0f);
                        nn_cpu_set_float_value_linear(config->filters, config->scales, 1.0f);
                    }
                }
                nn_cpu_set_float_value_linear(config->output_length * config->batch, config->output, 0.0f);
                nn_cpu_set_float_value_linear(config->output_length * config->batch, config->delta, 0.0f);
                nn_cpu_set_float_value_linear(config->filters_weight_length, config->weight_update, 0.0f);
                nn_cpu_set_float_value_linear(config->filters, config->bias_update, 0.0f);
                if (config->batch_normalize) {
                    nn_cpu_set_float_value_linear(config->filters, config->scales_update, 0.0f);
                    nn_cpu_set_float_value_linear(config->filters, config->mean, 0.0f);
                    nn_cpu_set_float_value_linear(config->filters, config->mean_delta, 0.0f);
                    nn_cpu_set_float_value_linear(config->filters, config->variance, 0.0f);
                    nn_cpu_set_float_value_linear(config->filters, config->variance_delta, 0.0f);
                    nn_cpu_set_float_value_linear(config->output_length * config->batch, config->x, 0.0f);
                    nn_cpu_set_float_value_linear(config->output_length * config->batch, config->x_normalize, 0.0f);
                }
                break;
            }
            case NN_CONNECTED:{
                nn_cpu_connected_layer* config = (nn_cpu_connected_layer*)start_layer;
                if (start_weight_layer->index_layer == i){
                    weight_darknet_network_connected_info* weight_layer = start_weight_layer;
                    if (start_weight_layer->type_layer != NN_CONNECTED) {printf("ERROR CONNECTED WEIGHT INIT: %zu INDEX LAYER", i); goto exit_function; }
                    if (weight_layer->length_weight != config->output_length * config->input_length) {fprintf(stderr, "ERROR CONNECTED WEIGHT INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_weight, config->output_length * config->input_length); goto exit_function; }
                    if (weight_layer->length_biases != config->output_length) {fprintf(stderr,"ERROR CONNECTED BIAS INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_biases, config->output_length); goto exit_function; }

                    memcpy(config->weight, weight_layer->weights, weight_layer->length_weight * sizeof(float));
                    memcpy(config->bias, weight_layer->biases, weight_layer->length_biases * sizeof(float));
                    if (config->batch_normalize) {
                        if (weight_layer->length_rolling_mean != config->output_length) {fprintf(stderr,"ERROR CONNECTED ROLLING MEAN INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_rolling_mean, config->output_length); goto exit_function; }
                        if (weight_layer->length_rolling_variance != config->output_length) {fprintf(stderr,"ERROR CONNECTED ROLLING VARIANCE INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_rolling_variance, config->output_length); goto exit_function; }
                        if (weight_layer->length_scales != config->output_length) {fprintf(stderr,"ERROR CONNECTED SCALES INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_scales, config->output_length); goto exit_function; }

                        memcpy(config->rolling_mean, weight_layer->rolling_mean, weight_layer->length_rolling_mean * sizeof(float));
                        memcpy(config->rolling_variance, weight_layer->rolling_variance, weight_layer->length_rolling_variance * sizeof(float));
                        memcpy(config->scales, weight_layer->scales, weight_layer->length_scales * sizeof(float));
                    }
                    start_weight_layer = start_weight_layer->next_layer;
                }else{
                    nn_cpu_set_random_float_linear(config->output_length * config->input_length, config->weight, 1, 0.0f);
                    nn_cpu_set_float_value_linear(config->output_length, config->bias, 0.0f);
                    if (config->batch_normalize) {
                        nn_cpu_set_float_value_linear(config->output_length, config->rolling_mean, 0.0f);
                        nn_cpu_set_float_value_linear(config->output_length, config->rolling_variance, 0.0f);
                        nn_cpu_set_float_value_linear(config->output_length, config->scales, 1.0f);
                    }
                }
                nn_cpu_set_float_value_linear(config->output_length * config->batch, config->output, 0.0f);
                nn_cpu_set_float_value_linear(config->output_length * config->batch, config->delta, 0.0f);
                nn_cpu_set_float_value_linear(config->output_length * config->input_length, config->weight_update, 0.0f);
                nn_cpu_set_float_value_linear(config->output_length, config->bias_update, 0.0f);
                if (config->batch_normalize){
                    nn_cpu_set_float_value_linear(config->output_length, config->scales_update, 0.0f);
                    nn_cpu_set_float_value_linear(config->output_length, config->mean, 0.0f);
                    nn_cpu_set_float_value_linear(config->output_length, config->mean_delta, 0.0f);
                    nn_cpu_set_float_value_linear(config->output_length, config->variance, 0.0f);
                    nn_cpu_set_float_value_linear(config->output_length, config->variance_delta, 0.0f);
                    nn_cpu_set_float_value_linear(config->output_length * config->batch, config->x, 0.0f);
                    nn_cpu_set_float_value_linear(config->output_length * config->batch, config->x_normalize, 0.0f);
                }
                break;
            }
            case NN_LOCAL:{
                nn_cpu_local_layer* config = (nn_cpu_local_layer*)start_layer;
                float scale = sqrtf(2.0f / (config->filter_length*config->filter_length*config->output_channels));
                if (start_weight_layer->index_layer == i){
                    weight_darknet_network_local_info* weight_layer = start_weight_layer;
                    if (start_weight_layer->type_layer != NN_LOCAL) {printf("ERROR LOCAL WEIGHT INIT: %zu INDEX LAYER", i); goto exit_function; }
                    if (weight_layer->length_weight != config->output_image_length * config->filters * config->filter_length * config->filter_length) {fprintf(stderr, "ERROR LOCAL WEIGHT INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_weight, config->output_image_length * config->filters * config->filter_length * config->filter_length); goto exit_function; }
                    if (weight_layer->length_biases != config->output_image_length) {fprintf(stderr,"ERROR LOCAL BIAS INIT: %zu INDEX LAYER; %zu != %d", i, weight_layer->length_biases, config->output_image_length); goto exit_function; }
                    memcpy(config->weight, weight_layer->weights, weight_layer->length_weight * sizeof(float));
                    memcpy(config->bias, weight_layer->biases, weight_layer->length_biases * sizeof(float));
                    start_weight_layer = start_weight_layer->next_layer;
                }else{
                    nn_cpu_set_random_float_linear(config->output_image_length * config->filters * config->filter_length * config->filter_length, config->weight, scale, 0.0f);
                    nn_cpu_set_float_value_linear(config->output_image_length, config->bias, 0.0f);
                }
                nn_cpu_set_float_value_linear(config->output_image_length * config->filters * config->filter_length * config->filter_length, config->weight_update, 0.0f);
                nn_cpu_set_float_value_linear(config->output_image_length, config->bias_update, 0.0f);
                nn_cpu_set_float_value_linear(config->batch * config->filters * config->output_height * config->output_width, config->output, 0.0f);
                nn_cpu_set_float_value_linear(config->batch * config->filters * config->output_height * config->output_width, config->delta, 0.0f);
                break;
            }
            case NN_MAXPOOL: {
                nn_cpu_maxpool_layer* config = (nn_cpu_maxpool_layer*)start_layer;
                nn_cpu_set_int_value_linear( config->batch * config->output_image_length, config->indices, 0);
                nn_cpu_set_float_value_linear( config->batch * config->output_image_length, config->output, 0.0f);
                nn_cpu_set_float_value_linear( config->batch * config->output_image_length, config->delta, 0.0f);
                break;
            }
            case NN_REGION: {
                nn_cpu_region_layer* config = (nn_cpu_region_layer*)start_layer;
                nn_cpu_set_float_value_linear(config->num * 2, config->bias, 0.5f);
                nn_cpu_set_float_value_linear(config->num * 2, config->bias_update, 0.0f);
                nn_cpu_set_float_value_linear(config->num * config->height * config->width * config->num * (config->classes + config->coords + 1), config->delta, 0.0f);
                nn_cpu_set_float_value_linear(config->num * config->height * config->width * config->num * (config->classes + config->coords + 1), config->output, 0.0f);
                break;
            }
                //case CFG_PARSER_TYPE_BATCHNORM:
                //    break;
            default: {

                break;
            }
            case 0:
                exit_function:;
                i = network->number_layers;
                break;
        }
    }
}
nn_cpu_network* nn_cpu_network_malloc(nn_network_config* config_network, nn_network_weight* weight_network){
    size_t size_layers = get_size_network(config_network);
    size_t size_parameters_network = config_network->number_layers * sizeof(float) + config_network
            ->number_layers * sizeof(float) + sizeof(nn_cpu_network);
    nn_cpu_network* network = (nn_cpu_network*)malloc(size_layers + size_parameters_network);
    init_pointers(config_network, network, size_parameters_network);
    init_network(config_network, network);
    init_network_layers(network, weight_network);
    nn_cpu_create_convolution_layers_images_ppm(network, 1);
    return network;
}



void nn_cpu_network_free(nn_cpu_network* network) {
    if(network) free(network);
    network = NULL;
}
//----------------------------------------------------------

void calc_network_cost(nn_cpu_network *network)
{
    float sum_cost = 0;
    unsigned number_cost = 0;
    nn_cpu_layer* start_layer = network->layers;
    for(unsigned i = 0; i < network->number_layers; ++i){
        switch (start_layer->type_layer) {
            case NN_REGION: sum_cost += ((nn_cpu_region_layer*)start_layer)->cost; number_cost++; break;
        }
    }
    network->cost = sum_cost / number_cost;
}

void nn_cpu_forward_network(nn_cpu_network *network)
{
    nn_cpu_layer* start_layer = network->layers;
    for(unsigned i = 0; i < network->number_layers; i++){
        start_layer->forward(start_layer, network);
        network->input = start_layer->output;
      //  if(start_layer->truth) {
      //      network->truth = start_layer->output;
      //  }
        start_layer = start_layer->next_layer;
    }
    calc_network_cost(network);
}
/*
float *network_predict(nn_cpu_network *net, float *input)
{
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    return  net->output;
}
*/