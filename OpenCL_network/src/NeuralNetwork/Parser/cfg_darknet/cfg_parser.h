//
// Created by human on 13.03.2020.
//

#ifndef OPENCL_CFG_PARSER_H
#define OPENCL_CFG_PARSER_H
#include "cfg_reader.h"
#include "../../nn_layers.h"
typedef enum {
    CFG_PARSER_TYPE_DECONVOLUTION = 1,
    CFG_PARSER_TYPE_CONVOLUTION,
    CFG_PARSER_TYPE_MAXPOOL,
    CFG_PARSER_TYPE_DROPOUT,
    CFG_PARSER_TYPE_LOCAL,
    CFG_PARSER_TYPE_CONNECTED,
    CFG_PARSER_TYPE_DETECTION,
    CFG_PARSER_TYPE_REGION
}CFG_PARSER_TYPE_LAYER;
struct cfg_darknet_layer_info;
typedef struct cfg_darknet_layer_info cfg_darknet_layer_info;
struct cfg_darknet_layer_info{
    cfg_darknet_layer_info *next_layer;
    CFG_PARSER_TYPE_LAYER type_layer;
    unsigned workspace_size;
    void *data;
};
typedef struct {
    cfg_darknet_layer_info* next_layer;
    CFG_PARSER_TYPE_LAYER type_layer;
    unsigned workspace_size;

    unsigned width;
    unsigned height;
    unsigned channels;
    unsigned batch;

    unsigned filters;
    unsigned filter_length;
    unsigned weight_length;
    unsigned stride;
    unsigned padding;
    unsigned groups;
    unsigned batch_normalize;
    unsigned transposed;

    unsigned output_width;
    unsigned output_height;
    unsigned output_channels;

    NN_ACTIVATION_FUNCTION function;
}cfg_darknet_convolution_layer_info;

typedef struct{
    cfg_darknet_layer_info* next_layer;
    CFG_PARSER_TYPE_LAYER type_layer;
    unsigned workspace_size;

    unsigned width;
    unsigned height;
    unsigned channels;
    unsigned batch;
    unsigned input_image_length;

    unsigned stride;
    unsigned filter_size;
    unsigned padding;
    unsigned output_width;
    unsigned output_height;
    unsigned output_channels;
    unsigned output_image_length;
    unsigned output_batch_length;

}cfg_darknet_maxpool_layer_info;

typedef struct{
    cfg_darknet_layer_info* next_layer;
    CFG_PARSER_TYPE_LAYER type_layer;
    unsigned workspace_size;
    unsigned output_image_length;
    unsigned input_image_length;

    unsigned batch;
    unsigned field_size;
    unsigned output_width;
    unsigned output_height;
    unsigned output_channels;
    float scale;
    float probability;

}cfg_darknet_dropout_layer_info;

typedef struct{
    cfg_darknet_layer_info* next_layer;
    CFG_PARSER_TYPE_LAYER type_layer;
    unsigned workspace_size;

    unsigned input_width;
    unsigned input_height;
    unsigned input_channels;
    unsigned batch;

    unsigned filters;
    unsigned filter_size;
    unsigned stride;
    unsigned pad;

    unsigned output_height;
    unsigned output_width;
    unsigned output_channels;

    unsigned input_image_length;
    unsigned output_image_length;

    NN_ACTIVATION_FUNCTION function;
}cfg_darknet_local_layer_info;

typedef struct{
    cfg_darknet_layer_info* next_layer;
    CFG_PARSER_TYPE_LAYER type_layer;
    unsigned workspace_size;

    unsigned width;
    unsigned height;
    unsigned input_channels;
    unsigned batch;

    unsigned batch_normalize;


    unsigned output_height;
    unsigned output_width;
    unsigned output_channels;

    unsigned input_length;
    unsigned output_length;

    //unsigned filters;
    //unsigned filter_size;
    unsigned stride;

    NN_ACTIVATION_FUNCTION function;
}cfg_darknet_connected_layer_info;

typedef struct{
    cfg_darknet_layer_info* next_layer;
    CFG_PARSER_TYPE_LAYER type_layer;
    unsigned workspace_size;

    unsigned batch;
    unsigned input;
    unsigned output;
    unsigned classes;
    unsigned coords;

    unsigned rescore;
    unsigned side;
    unsigned n;

    unsigned softmax;
    unsigned sqrt;

    unsigned max_boxes;
    unsigned forced;
    unsigned random;
    unsigned reorg;
    float coord_scale;
    float object_scale;
    float noobject_scale;
    float class_scale;
    float jitter;

    NN_ACTIVATION_FUNCTION function;
}cfg_darknet_detection_layer_info;

typedef struct{
    cfg_darknet_layer_info* next_layer;
    CFG_PARSER_TYPE_LAYER type_layer;
    unsigned workspace_size;

    unsigned width;
    unsigned height;
    unsigned channels;
    unsigned batch;

    unsigned output_height;
    unsigned output_width;
    unsigned output_channels;

    unsigned truths;
    unsigned coords;
    unsigned classes;
    unsigned num;
    unsigned log;
    unsigned sqrt;
    unsigned softmax;
    unsigned background;
    unsigned max_boxes;
    float jitter;
    unsigned rescore;
    float thresh;
    unsigned classfix;
    unsigned absolute;
    unsigned random;
    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    unsigned bias_match;


    unsigned number_biases;
    float biases[0];
}cfg_darknet_region_layer_info;

typedef struct {
    unsigned number_layers;
    int batch;
    int subdivisions; //batch division
    int width;
    int height;
    int channels;
    int burn_in;
    int max_batches;
    int policy;
    float momentum;
    float decay;
    float angle;
    float saturation;
    float exposure;
    float hue;
    float learning_rate;
    unsigned number_steps;
    int* steps;
    float* scales_or_gamma; //policy

    unsigned max_workspace_size;
    cfg_darknet_layer_info* layers;
}cfg_darknet_network_info;
NN_LAYER_TYPE get_layer_type(const char* name_type);
cfg_darknet_network_info* nn_darknet_network_parser_malloc(struct cfg_config* options);
void nn_darknet_network_parser_free(cfg_darknet_network_info* cfg_net);
#endif //OPENCL_CFG_PARSER_H
