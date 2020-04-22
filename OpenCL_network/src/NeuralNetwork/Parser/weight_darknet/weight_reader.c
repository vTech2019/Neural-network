//
// Created by human on 15.03.2020.
//

#include "../../File/file.h"
#include "weight_reader.h"
#include "../cfg_darknet/cfg_parser.h"
#include "../cfg_net.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void transpose_matrix(float *a, unsigned rows, unsigned cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    unsigned x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

char* load_connected_weights(weight_darknet_network_connected_info* weight_layer, cfg_darknet_connected_layer_info* cfg_layer, FILE *fp, int transpose, size_t index_layer)
{
    char* current_ptr = ((char*)weight_layer) + sizeof(weight_darknet_network_connected_info);
    weight_layer->type_layer = NN_CONNECTED;
    weight_layer->index_layer = index_layer;
    weight_layer->length_weight = cfg_layer->output_channels * cfg_layer->input_channels;
    weight_layer->length_biases = cfg_layer->output_channels;
    weight_layer->length_scales = 0;
    weight_layer->length_rolling_mean = 0;
    weight_layer->length_rolling_variance = 0;
    weight_layer->biases = (float *) current_ptr;
    current_ptr += weight_layer->length_biases * sizeof(float);
    fread(weight_layer->biases, sizeof(float), weight_layer->length_biases, fp);
    if (cfg_layer->batch_normalize){
        weight_layer->length_scales = cfg_layer->output_channels;
        weight_layer->length_rolling_mean = cfg_layer->output_channels;
        weight_layer->length_rolling_variance = cfg_layer->output_channels;
        weight_layer->scales = current_ptr;
        current_ptr += weight_layer->length_scales * sizeof(float);
        weight_layer->rolling_mean = current_ptr;
        current_ptr += weight_layer->length_rolling_mean * sizeof(float);
        weight_layer->rolling_variance = current_ptr;
        current_ptr += weight_layer->length_rolling_variance * sizeof(float);
        fread(weight_layer->scales, sizeof(float), weight_layer->length_scales, fp);
        fread(weight_layer->rolling_mean, sizeof(float), weight_layer->length_rolling_mean, fp);
        fread(weight_layer->rolling_variance, sizeof(float), weight_layer->length_rolling_variance, fp);
    }
    weight_layer->weights = (float *) current_ptr;
    current_ptr += weight_layer->length_weight * sizeof(float);

    fread(weight_layer->weights, sizeof(float), weight_layer->length_weight, fp);
    if(transpose){
        transpose_matrix(weight_layer->weights, cfg_layer->input_channels, cfg_layer->input_channels);
    }
    weight_layer->next_layer = (cfg_darknet_weight_layer_info *) current_ptr;
    return current_ptr;
}

/*void load_batchnorm_weights(weight_darknet_network_batchnorm_info* weight_layer, cfg_darknet_batchnorm_layer_info* cfg_layer, FILE *fp)
{
    fread(weight_layer->scales, sizeof(float), cfg_layer->channels, fp);
    fread(weight_layer->rolling_mean, sizeof(float), cfg_layer->channels, fp);
    fread(weight_layer->rolling_variance, sizeof(float), cfg_layer->channels, fp);
}*/


char* load_local_weights(weight_darknet_network_local_info* weight_layer, cfg_darknet_local_layer_info* cfg_layer, FILE *fp, size_t index_layer)
{
    char* current_ptr = ((char*)weight_layer) + sizeof(weight_darknet_network_local_info);
    weight_layer->type_layer = NN_CONNECTED;
    weight_layer->index_layer = index_layer;
    weight_layer->length_weight += cfg_layer->output_image_length * cfg_layer->filters * cfg_layer->filter_size * cfg_layer->filter_size;
    weight_layer->length_biases += cfg_layer->output_image_length * cfg_layer->filters;
    weight_layer->biases = current_ptr;
    current_ptr += weight_layer->length_biases * sizeof(float);
    weight_layer->weights = current_ptr;
    current_ptr += weight_layer->length_weight * sizeof(float);
    fread(weight_layer->biases, sizeof(float), weight_layer->length_biases, fp);
    fread(weight_layer->weights, sizeof(float), weight_layer->length_weight, fp);
    weight_layer->next_layer = (cfg_darknet_weight_layer_info *) current_ptr;
    return current_ptr;
}
char* load_convolutional_weights(weight_darknet_network_convolution_info* weight_layer, cfg_darknet_convolution_layer_info* cfg_layer, FILE *fp, size_t index_layer)
{
    char* current_ptr = ((char*)weight_layer) + sizeof(weight_darknet_network_convolution_info);
    weight_layer->type_layer = NN_CONVOLUTIONAL;
    weight_layer->index_layer = index_layer;
    weight_layer->length_biases = cfg_layer->filters;
    weight_layer->length_weight = cfg_layer->channels / cfg_layer->groups * cfg_layer->filters * cfg_layer->filter_length * cfg_layer->filter_length;
    weight_layer->length_scales = 0;
    weight_layer->length_rolling_mean = 0;
    weight_layer->length_rolling_variance = 0;
    weight_layer->biases = (float *) current_ptr;
    current_ptr += weight_layer->length_biases * sizeof(float);
    fread(weight_layer->biases, sizeof(float), weight_layer->length_biases, fp);
    if (cfg_layer->batch_normalize){
        weight_layer->length_scales = cfg_layer->filters;
        weight_layer->length_rolling_mean = cfg_layer->filters;
        weight_layer->length_rolling_variance = cfg_layer->filters;
        weight_layer->scales = (float *) current_ptr;
        current_ptr += cfg_layer->filters * sizeof(float);
        weight_layer->rolling_mean = (float *) current_ptr;
        current_ptr += cfg_layer->filters * sizeof(float);
        weight_layer->rolling_variance = (float *) current_ptr;
        current_ptr += cfg_layer->filters * sizeof(float);
        fread(weight_layer->scales, sizeof(float), weight_layer->length_scales, fp);
        fread(weight_layer->rolling_mean, sizeof(float), weight_layer->length_rolling_mean, fp);
        fread(weight_layer->rolling_variance, sizeof(float), weight_layer->length_rolling_variance, fp);
    }
    weight_layer->weights = (float *) current_ptr;
    current_ptr += weight_layer->length_weight * sizeof(float);
    fread(weight_layer->weights, sizeof(float), weight_layer->length_weight, fp);
    if (cfg_layer->transposed) {
        transpose_matrix(weight_layer->weights, cfg_layer->channels*cfg_layer->filter_length*cfg_layer->filter_length, cfg_layer->filters);
    }
    weight_layer->next_layer = (cfg_darknet_weight_layer_info *) current_ptr;
    return current_ptr;
}
size_t get_weight_size(cfg_darknet_network_info* cfg){
    size_t data_size = 0;
    cfg_darknet_layer_info* start_layer = cfg->layers;
    for (size_t i = 0; i < cfg->number_layers; i++) {
        switch (start_layer->type_layer){
            case CFG_PARSER_TYPE_CONVOLUTION:{
                cfg_darknet_convolution_layer_info* layer = (cfg_darknet_convolution_layer_info*)start_layer;
                data_size += sizeof(weight_darknet_network_convolution_info);
                data_size += layer->filters * sizeof(float);
                data_size += layer->channels / layer->groups * layer->filters * layer->filter_length * layer->filter_length * sizeof(float);
                if (layer->batch_normalize){
                    data_size += 3 * layer->filters * sizeof(float);
                }
                break;
            }
            case CFG_PARSER_TYPE_DECONVOLUTION:{
                /*cfg_darknet_deconvolution_layer_info* layer = (cfg_darknet_deconvolution_layer_info*)start_layer;
                data_size += layer->filters * sizeof(float);
                data_size += layer->channels / layer->groups * layer->filters * layer->filter_size * layer->filter_size * sizeof(float);
                if (layer->batch_normalize){
                    data_size += 3 * layer->filters * sizeof(float);
                }
                data_size += sizeof(weight_darknet_network_deconvolution_info);*/
                break;
            }
            case CFG_PARSER_TYPE_CONNECTED:{
                cfg_darknet_connected_layer_info* layer = (cfg_darknet_connected_layer_info*)start_layer;
                data_size += sizeof(weight_darknet_network_connected_info);
                data_size += layer->output_width * layer->output_height * layer->output_channels * sizeof(float);
                data_size += layer->width * layer->height * layer->input_channels * layer->output_height * layer->output_width * layer->output_channels * sizeof(float);
                if (layer->batch_normalize) {
                    data_size += 3 * layer->output_height * layer->output_width * layer->output_channels * sizeof(float);
                }
                break;
            }
            case CFG_PARSER_TYPE_LOCAL:{
                cfg_darknet_local_layer_info* layer = (cfg_darknet_local_layer_info*)start_layer;
                data_size += sizeof(weight_darknet_network_local_info);
                data_size += layer->output_image_length * layer->filters * layer->filter_size * layer->filter_size * sizeof(float);
                data_size += layer->output_image_length * layer->filters * sizeof(float);
                break;
            }
            //case CFG_PARSER_TYPE_BATCHNORM:
                /*cfg_darknet_batchnorm_layer_info* layer = (cfg_darknet_batchnorm_layer_info*)start_layer;
                data_size += layer->channels * sizeof(float);
                data_size += layer->channels * sizeof(float);
                data_size += layer->channels * sizeof(float);*/
            //    break;
            default:
                break;
        }
        start_layer = start_layer->next_layer;
    }
    return data_size;
}
void load_weights(cfg_darknet_network_info* cfg, weight_darknet_network_info* weight, FILE* fp, int transpose) {
    weight->current_layer = (cfg_darknet_weight_layer_info *) (((char *) weight) + sizeof(weight_darknet_network_info));
    weight_darknet_network_convolution_info* start_weight = weight->current_layer;
    cfg_darknet_layer_info* start_layer = cfg->layers;
    for (size_t i = 0; i < cfg->number_layers; i++) {
        switch (start_layer->type_layer){
            case CFG_PARSER_TYPE_DECONVOLUTION:
                break;
            case CFG_PARSER_TYPE_CONVOLUTION:
            {
                cfg_darknet_convolution_layer_info* c_layer = (cfg_darknet_convolution_layer_info*)start_layer;
                weight_darknet_network_convolution_info* w_layer = (weight_darknet_network_convolution_info*)start_weight;
                start_weight = load_convolutional_weights(w_layer, c_layer, fp, i);
                break;
            }
            case CFG_PARSER_TYPE_CONNECTED:{
                cfg_darknet_connected_layer_info* c_layer = (cfg_darknet_connected_layer_info*)start_layer;
                weight_darknet_network_connected_info* w_layer = (weight_darknet_network_connected_info*)start_weight;
                start_weight = load_connected_weights(w_layer, c_layer, fp, transpose, i);
                break;
            }
            case CFG_PARSER_TYPE_LOCAL:{
                cfg_darknet_local_layer_info* c_layer = (cfg_darknet_local_layer_info*)start_layer;
                weight_darknet_network_local_info* w_layer = (weight_darknet_network_local_info*)start_weight;
                start_weight = load_local_weights(w_layer, c_layer, fp, i);
                break;
            }
            //case CFG_PARSER_TYPE_BATCHNORM:
            //    break;
            default: break;
        }
        start_layer = start_layer->next_layer;
    }
}

weight_darknet_network_info* weights_reader_malloc(cfg_darknet_network_info* cfg, char *filename)
{
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    size_t weight_size = get_weight_size(cfg);

    weight_darknet_network_info *net = calloc(1, weight_size + sizeof(weight_darknet_network_info));
    if (net){
        net->size_data_malloc = weight_size + sizeof(weight_darknet_network_info);
        int major;
        int minor;
        int revision;
        fread(&major, sizeof(int), 1, fp);
        fread(&minor, sizeof(int), 1, fp);
        fread(&revision, sizeof(int), 1, fp);
        if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
            fread(&net->seen, sizeof(size_t), 1, fp);
        } else {
            int iseen = 0;
            fread(&iseen, sizeof(int), 1, fp);
            net->seen = iseen;
        }
        int transpose = (major > 1000) || (minor > 1000);
        load_weights(cfg, net, fp, transpose);

        fprintf(stderr, "Done!\n");
    }else{

        fprintf(stderr, "Not done!\n");
    }

    fclose(fp);
    return net;
}

void weights_reader_free(weight_darknet_network_info* weight){
    if (weight) free(weight);
    weight = NULL;
}