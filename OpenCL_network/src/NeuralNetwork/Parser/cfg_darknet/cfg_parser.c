//
// Created by human on 13.03.2020.
//

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "cfg_parser.h"

typedef enum {
    NN_CONSTANT, NN_STEP, NN_EXP, NN_POLY, NN_STEPS, NN_SIG, NN_RANDOM
} LEARNING_RATE_POLICY;

LEARNING_RATE_POLICY get_policy(char *policy)
{
    if (strcmp(policy, "random")==0) return NN_RANDOM;
    if (strcmp(policy, "poly")==0) return NN_POLY;
    if (strcmp(policy, "constant")==0) return NN_CONSTANT;
    if (strcmp(policy, "step")==0) return NN_STEP;
    if (strcmp(policy, "exp")==0) return NN_EXP;
    if (strcmp(policy, "sigmoid")==0) return NN_SIG;
    if (strcmp(policy, "steps")==0) return NN_STEPS;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", policy);
    return NN_CONSTANT;
}

void init_policy(cfg_darknet_network_info* net_info, struct cfg_type_object* layer_info)
{
    net_info->policy = get_policy(cfg_option_find_string(layer_info, "policy", "constant"));
    switch(net_info->policy){
        case NN_STEP:
            cfg_option_find_floats_return_number_write(layer_info, "step", net_info->scales_or_gamma, net_info->number_steps);
            cfg_option_find_ints_return_number_write(layer_info, "scale", net_info->steps, net_info->number_steps);
            break;
        case NN_STEPS:
            cfg_option_find_floats_return_number_write(layer_info, "steps", net_info->scales_or_gamma, net_info->number_steps);
            cfg_option_find_ints_return_number_write(layer_info, "scales", net_info->steps, net_info->number_steps);
            break;
        case NN_EXP:
            cfg_option_find_floats_return_number_write(layer_info, "gamma", net_info->scales_or_gamma, net_info->number_steps);
            break;
        case NN_SIG:
            cfg_option_find_floats_return_number_write(layer_info, "gamma", net_info->scales_or_gamma, net_info->number_steps);
            cfg_option_find_ints_return_number_write(layer_info, "step", net_info->steps, net_info->number_steps);
            break;
        case NN_CONSTANT:
        case NN_POLY:
        case NN_RANDOM:
            break;
    }
}

NN_LAYER_TYPE get_layer_type(const char* name_type)
{
    if (strcmp(name_type, "active") == 0)           return NN_ACTIVE;
    if (strcmp(name_type, "avg") == 0)              return NN_AVGPOOL;
    if (strcmp(name_type, "avgpool") == 0)          return NN_AVGPOOL;
    if (strcmp(name_type, "batchnorm") == 0)        return NN_BATCHNORM;
    if (strcmp(name_type, "blank") == 0)            return NN_BLANK;
    if (strcmp(name_type, "conn") == 0)             return NN_CONNECTED;
    if (strcmp(name_type, "connected") == 0)        return NN_CONNECTED;
    if (strcmp(name_type, "conv") == 0)             return NN_CONVOLUTIONAL;
    if (strcmp(name_type, "convolutional") == 0)    return NN_CONVOLUTIONAL;
    if (strcmp(name_type, "cost") == 0)             return NN_COST;
    if (strcmp(name_type, "crnn") == 0)             return NN_CRNN;
    if (strcmp(name_type, "crop") == 0)             return NN_CROP;
    if (strcmp(name_type, "deconvolutional") == 0)  return NN_DECONVOLUTIONAL;
    if (strcmp(name_type, "deconv") == 0)           return NN_DECONVOLUTIONAL;
    if (strcmp(name_type, "detection") == 0)        return NN_DETECTION;
    if (strcmp(name_type, "dropout") == 0)          return NN_DROPOUT;
    if (strcmp(name_type, "gru") == 0)              return NN_GRU;
    if (strcmp(name_type, "iseg") == 0)             return NN_ISEG;
    if (strcmp(name_type, "l2norm") == 0)           return NN_L2NORM;
    if (strcmp(name_type, "local") == 0)            return NN_LOCAL;
    if (strcmp(name_type, "logxent") == 0)          return NN_LOGXENT;
    if (strcmp(name_type, "lstm") == 0)             return NN_LSTM;
    if (strcmp(name_type, "max") == 0)              return NN_MAXPOOL;
    if (strcmp(name_type, "maxpool") == 0)          return NN_MAXPOOL;
    if (strcmp(name_type, "network") == 0)          return NN_NETWORK;
    if (strcmp(name_type, "net") == 0)              return NN_NETWORK;
    if (strcmp(name_type, "lrn") == 0)              return NN_NORMALIZATION;
    if (strcmp(name_type, "normalization") == 0)    return NN_NORMALIZATION;
    if (strcmp(name_type, "region") == 0)           return NN_REGION;
    if (strcmp(name_type, "reorg") == 0)            return NN_REORG;
    if (strcmp(name_type, "rnn") == 0)              return NN_RNN;
    if (strcmp(name_type, "route") == 0)            return NN_ROUTE;
    if (strcmp(name_type, "shortcut") == 0)         return NN_SHORTCUT;
    if (strcmp(name_type, "soft") == 0)             return NN_SOFTMAX;
    if (strcmp(name_type, "softmax") == 0)          return NN_SOFTMAX;
    if (strcmp(name_type, "upsample") == 0)         return NN_UPSAMPLE;
    if (strcmp(name_type, "xnor") == 0)             return NN_XNOR;
    if (strcmp(name_type, "yolo") == 0)             return NN_YOLO;
    fprintf(stderr, "Couldn't find type %s\n", name_type);
    return 0;
}


NN_ACTIVATION_FUNCTION get_activation_type(const char* name_function)
{
    if (strcmp(name_function, "elu") == 0) return NN_ELU;
    if (strcmp(name_function, "hardtan") == 0) return NN_HARDTAN;
    if (strcmp(name_function, "leaky") == 0) return NN_LEAKY;
    if (strcmp(name_function, "lhtan") == 0) return NN_LHTAN;
    if (strcmp(name_function, "linear") == 0) return NN_LINEAR;
    if (strcmp(name_function, "loggy") == 0) return NN_LOGGY;
    if (strcmp(name_function, "logistic") == 0) return NN_LOGISTIC;
    if (strcmp(name_function, "plse") == 0) return NN_PLSE;
    if (strcmp(name_function, "ramp") == 0) return NN_RAMP;
    if (strcmp(name_function, "relie") == 0) return NN_RELIE;
    if (strcmp(name_function, "relu") == 0) return NN_RELU;
    if (strcmp(name_function, "selu") == 0) return NN_SELU;
    if (strcmp(name_function, "stair") == 0) return NN_STAIR;
    if (strcmp(name_function, "tanh") == 0) return NN_TANH;
    fprintf(stderr, "Couldn't find activation function %s\n", name_function);
    return 0;
}
struct cfg_type_object* nn_darknet_get_net_option_layer(struct cfg_config* options){
    struct cfg_type_object* object = options->objects;
    for (size_t i = 0; i < options->number_options; i++) {
        if (get_layer_type(object->current_type_object_value->value) == NN_NETWORK)
            return object;
        object = object->next_type_object;
    }
    return NULL;
}

void nn_darknet_layer_convolutional_parser(cfg_darknet_convolution_layer_info* info, struct cfg_type_object* layer_info, unsigned width, unsigned height, unsigned channels, unsigned batch)
{
    int pad = cfg_option_find_int(layer_info, "pad",0);
    info->type_layer = CFG_PARSER_TYPE_CONVOLUTION;
    info->width = width;
    info->height = height;
    info->channels = channels;
    info->batch = batch;
    info->filters = cfg_option_find_int(layer_info, "filters",1);
    info->filter_length = cfg_option_find_int(layer_info, "size",1);
    info->stride = cfg_option_find_int(layer_info, "stride",1);
    info->padding = cfg_option_find_int(layer_info, "padding",0);
    info->groups = cfg_option_find_int(layer_info, "groups", 1);
    info->batch_normalize = cfg_option_find_int(layer_info, "batch_normalize", 0);
    info->function = get_activation_type(cfg_option_find_string(layer_info, "activation", "logistic"));
    info->transposed = cfg_option_find_int(layer_info, "flipped", 0);

    if(pad) info->padding = info->filter_length / 2;
    info->output_width = (width + 2*info->padding - info->filter_length) / info->stride + 1;
    info->output_height = (height + 2*info->padding - info->filter_length) / info->stride + 1;
    info->output_channels = info->filters;

    info->weight_length = info->channels / info->groups * info->filters * info->filter_length * info->filter_length;

    info->workspace_size = info->output_height * info->output_width * info->filter_length * info->filter_length * info->channels / info->groups * sizeof(float);
    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", info->filters, info->filter_length, info->filter_length, info->stride, width, height, channels, info->output_width, info->output_height, info->output_channels, (2.0 * info->filters * info->filter_length*info->filter_length*channels/info->groups * info->output_height*info->output_width)/1000000000.);

}

void nn_darknet_layer_maxpool_parser(cfg_darknet_maxpool_layer_info* info, struct cfg_type_object* layer_info, unsigned width, unsigned height, unsigned channels, unsigned batch)
{
    info->type_layer = CFG_PARSER_TYPE_MAXPOOL;
    info->stride = cfg_option_find_int(layer_info, "stride",1);
    info->filter_size = cfg_option_find_int(layer_info, "size", (int)info->stride);
    info->padding = cfg_option_find_int(layer_info, "padding", (int)info->filter_size - 1);
    info->width = width;
    info->height = height;
    info->channels = channels;
    info->batch = batch;
    info->output_width = (width + info->padding - info->filter_size)/info->stride + 1;
    info->output_height = (height + info->padding - info->filter_size)/info->stride + 1;
    info->output_channels = channels;
    info->input_image_length = height * width * channels;
    info->output_image_length = info->output_width * info->output_height * info->channels;
    info->output_batch_length = info->output_image_length * batch;

    info->workspace_size = info->output_image_length * sizeof(float);

    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", info->filter_size, info->filter_size, info->stride, width, height, channels, info->output_width, info->output_height, info->output_channels);
}

void nn_darknet_layer_dropout_parser(cfg_darknet_dropout_layer_info* info, struct cfg_type_object* layer_info, unsigned width, unsigned height, unsigned channels, unsigned batch)
{
    info->type_layer = CFG_PARSER_TYPE_DROPOUT;
    info->batch = batch;
    info->input_image_length = width * height * channels;
    info->output_image_length = width * height * channels;
    info->field_size = info->input_image_length * batch;
    info->output_width = width;
    info->output_height = height;
    info->output_channels = channels;
    info->probability = cfg_option_find_float(layer_info, "probability", .5);
    info->scale = 1.0f/(1.0f - info->probability);

    info->workspace_size = info->input_image_length * sizeof(float);
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", info->probability, info->input_image_length, info->output_image_length);
}
void nn_darknet_layer_local_parser(cfg_darknet_local_layer_info* info, struct cfg_type_object* layer_info, unsigned width, unsigned height, unsigned channels, unsigned batch)
{
    info->type_layer = CFG_PARSER_TYPE_LOCAL;
    info->input_width = width;
    info->input_height = height;
    info->input_channels = channels;
    info->batch = batch;
    info->filters = cfg_option_find_int(layer_info, "filters",1);
    info->filter_size = cfg_option_find_int(layer_info, "size",1);
    info->stride = cfg_option_find_int(layer_info, "stride",1);
    info->pad = cfg_option_find_int(layer_info, "pad",0);
    info->function = get_activation_type(cfg_option_find_string(layer_info, "activation", "logistic"));
    info->output_height = (height - (!info->pad ? info->filter_size : 1)) / info->stride + 1;
    info->output_width = (width - (!info->pad ? info->filter_size : 1)) / info->stride + 1;
    info->output_channels = channels;

    info->output_image_length = info->output_height * info->output_width * channels;
    info->input_image_length = width * height * channels;

    info->workspace_size = info->output_height * info->output_width * info->filter_size * info->filter_size * channels * sizeof(float);
    fprintf(stderr, "Local Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", height, width, channels, info->filters, info->output_height, info->output_width, info->filters);
}
void nn_darknet_layer_connected_parser(cfg_darknet_connected_layer_info* info, struct cfg_type_object* layer_info, unsigned width, unsigned height, unsigned channels, unsigned batch)
{
    info->type_layer = CFG_PARSER_TYPE_CONNECTED;
    info->function = get_activation_type(cfg_option_find_string(layer_info, "activation", "logistic"));
    info->batch = batch;
    info->batch_normalize = cfg_option_find_int(layer_info, "batch_normalize", 0);
    info->height = 1;
    info->width = 1;
    info->output_width = 1;
    info->output_height = 1;
    info->output_channels = cfg_option_find_int(layer_info, "output",1);
    info->input_channels = width * height * channels;

    info->output_length = cfg_option_find_int(layer_info, "output",1);
    info->input_length = width * height * channels;

    info->workspace_size =  info->output_length * sizeof(float);
    fprintf(stderr, "connected                            %4d  ->  %4d\n", info->input_channels, info->output_length);
}

void nn_darknet_layer_region_parser(cfg_darknet_region_layer_info* info, struct cfg_type_object* layer_info, unsigned width, unsigned height, unsigned batch)
{
    info->type_layer = CFG_PARSER_TYPE_REGION;
    info->coords = cfg_option_find_int(layer_info, "coords", 4);
    info->classes = cfg_option_find_int(layer_info, "classes", 20);
    info->num = cfg_option_find_int(layer_info, "num", 1);
    info->batch = batch;

    info->width = width;
    info->height = height;
    info->channels = info->num * (info->classes + info->coords + 1);

    info->output_height = height;
    info->output_width = width;
    info->output_channels = info->channels;

    //assert(height * width * info->channels == width * height * channels);

    info->log = cfg_option_find_int(layer_info, "log", 0);
    info->sqrt = cfg_option_find_int(layer_info, "sqrt", 0);

    info->softmax = cfg_option_find_int(layer_info, "softmax", 0);
    info->background = cfg_option_find_int(layer_info, "background", 0);
    info->max_boxes = cfg_option_find_int(layer_info, "max",30);
    info->jitter = cfg_option_find_float(layer_info, "jitter", .2);
    info->rescore = cfg_option_find_int(layer_info, "rescore",0);

    info->thresh = cfg_option_find_float(layer_info, "thresh", .5);
    info->classfix = cfg_option_find_int(layer_info, "classfix", 0);
    info->absolute = cfg_option_find_int(layer_info, "absolute", 0);
    info->random = cfg_option_find_int(layer_info, "random", 0);

    info->coord_scale = cfg_option_find_float(layer_info, "coord_scale", 1);
    info->object_scale = cfg_option_find_float(layer_info, "object_scale", 1);
    info->noobject_scale = cfg_option_find_float(layer_info, "noobject_scale", 1);
    info->mask_scale = cfg_option_find_float(layer_info, "mask_scale", 1);
    info->class_scale = cfg_option_find_float(layer_info, "class_scale", 1);
    info->bias_match = cfg_option_find_int(layer_info, "bias_match",0);

    info->truths = 30*(info->coords + 1);
    //char *tree_file = option_find_str(layer_info, "tree", 0);
    //char *map_file = option_find_str(layer_info, "map", 0);
    info->number_biases = cfg_option_get_number_values(layer_info,  "anchors");
    cfg_option_find_floats_return_number_write(layer_info, "anchors", info->biases, info->number_biases);

    info->workspace_size = info->output_height * info->output_width * info->output_channels * sizeof(float);
    fprintf(stderr, "detection\n");
}

void nn_darknet_layer_detection_parser(cfg_darknet_detection_layer_info* info, struct cfg_type_object* layer_info, unsigned width, unsigned height, unsigned channels, unsigned batch)
{
    info->type_layer = CFG_PARSER_TYPE_DETECTION;
    info->batch = batch;
    info->input = width * height * channels;
    info->output = width * height * channels;
    info->coords = cfg_option_find_int(layer_info, "coords", 1);
    info->classes = cfg_option_find_int(layer_info, "classes", 1);
    info->rescore = cfg_option_find_int(layer_info, "rescore", 0);
    info->n = cfg_option_find_int(layer_info, "num", 1);
    info->side = cfg_option_find_int(layer_info, "side", 7);
    assert(info->side*info->side*((1 + info->coords)*info->n + info->classes) == info->input);
    info->softmax = cfg_option_find_int(layer_info, "softmax", 0);
    info->sqrt = cfg_option_find_int(layer_info, "sqrt", 0);
    info->max_boxes = cfg_option_find_int(layer_info, "max",90);
    info->random = cfg_option_find_int(layer_info, "random", 0);
    info->reorg = cfg_option_find_int(layer_info, "reorg", 0);
    info->forced = cfg_option_find_int(layer_info, "forced", 0);
    info->coord_scale = cfg_option_find_float(layer_info, "coord_scale", 1);
    info->object_scale = cfg_option_find_float(layer_info, "object_scale", 1);
    info->noobject_scale = cfg_option_find_float(layer_info, "noobject_scale", 1);
    info->class_scale = cfg_option_find_float(layer_info, "class_scale", 1);
    info->jitter = cfg_option_find_float(layer_info, "jitter", .2);

    info->workspace_size = info->output * sizeof(float);
    fprintf(stderr, "detection\n");
}
void nn_darknet_layer_parser(struct cfg_config* options, cfg_darknet_network_info* net_info){
    unsigned width = net_info->width;
    unsigned height = net_info->height;
    unsigned channels = net_info->channels;
    unsigned batch = net_info->batch;

    struct cfg_type_object* object = options->objects;
    fprintf(stderr, "layer     filters    size              input                output\n");
    for (size_t i = 0, offset = 0, last_offset = 0; i < options->number_options; i++) {
        switch(get_layer_type(object->current_type_object_value->value)){
            case NN_CONVOLUTIONAL:
                nn_darknet_layer_convolutional_parser((cfg_darknet_convolution_layer_info*)&((char*)net_info->layers)[offset], object, width, height, channels, batch);
                ((cfg_darknet_convolution_layer_info*)&((char*)net_info->layers)[offset])->next_layer = (cfg_darknet_layer_info *) (
                        (char *) net_info->layers + offset + sizeof(cfg_darknet_convolution_layer_info));
                height      = ((cfg_darknet_convolution_layer_info*)&((char*)net_info->layers)[offset])->output_height;
                width       = ((cfg_darknet_convolution_layer_info*)&((char*)net_info->layers)[offset])->output_width;
                channels    = ((cfg_darknet_convolution_layer_info*)&((char*)net_info->layers)[offset])->output_channels;
                offset += sizeof(cfg_darknet_convolution_layer_info);
                break;
            case NN_DECONVOLUTIONAL:
                break;
            case NN_CONNECTED:
                nn_darknet_layer_connected_parser((cfg_darknet_connected_layer_info*)&((char*)net_info->layers)[offset], object, width, height, channels, batch);
                ((cfg_darknet_connected_layer_info*)&((char*)net_info->layers)[offset])->next_layer = (cfg_darknet_layer_info *) (
                        (char *) net_info->layers + offset + sizeof(cfg_darknet_connected_layer_info));
                height      = ((cfg_darknet_connected_layer_info*)&((char*)net_info->layers)[offset])->output_height;
                width       = ((cfg_darknet_connected_layer_info*)&((char*)net_info->layers)[offset])->output_width;
                channels    = ((cfg_darknet_connected_layer_info*)&((char*)net_info->layers)[offset])->output_channels;
                offset += sizeof(cfg_darknet_connected_layer_info);
                break;
            case NN_MAXPOOL:
                nn_darknet_layer_maxpool_parser((cfg_darknet_maxpool_layer_info*)&((char*)net_info->layers)[offset], object, width, height, channels, batch);
                ((cfg_darknet_maxpool_layer_info*)&((char*)net_info->layers)[offset])->next_layer = (cfg_darknet_layer_info *) (
                        (char *) net_info->layers + offset + sizeof(cfg_darknet_maxpool_layer_info));
                height      = ((cfg_darknet_maxpool_layer_info*)&((char*)net_info->layers)[offset])->output_height;
                width       = ((cfg_darknet_maxpool_layer_info*)&((char*)net_info->layers)[offset])->output_width;
                channels    = ((cfg_darknet_maxpool_layer_info*)&((char*)net_info->layers)[offset])->output_channels;
                offset += sizeof(cfg_darknet_maxpool_layer_info);
                break;
            case NN_SOFTMAX:
                break;
            case NN_DETECTION:
                nn_darknet_layer_detection_parser((cfg_darknet_detection_layer_info*)&((char*)net_info->layers)[offset], object, width, height, channels, batch);
                ((cfg_darknet_detection_layer_info*)&((char*)net_info->layers)[offset])->next_layer = (cfg_darknet_layer_info *) (
                        ((char *) net_info->layers) + offset + sizeof(cfg_darknet_detection_layer_info));
                height      = 0;
                width       = 0;
                channels    = 0;
                offset += sizeof(cfg_darknet_detection_layer_info);
                break;
            case NN_DROPOUT:
                nn_darknet_layer_dropout_parser((cfg_darknet_dropout_layer_info*)&((char*)net_info->layers)[offset], object, width, height, channels, batch);
                ((cfg_darknet_dropout_layer_info*)&((char*)net_info->layers)[offset])->next_layer = (cfg_darknet_layer_info *) (
                        ((char *) net_info->layers) + offset + sizeof(cfg_darknet_dropout_layer_info));
                height      = ((cfg_darknet_dropout_layer_info*)&((char*)net_info->layers)[offset])->output_height;
                width       = ((cfg_darknet_dropout_layer_info*)&((char*)net_info->layers)[offset])->output_width;
                channels    = ((cfg_darknet_dropout_layer_info*)&((char*)net_info->layers)[offset])->output_channels;
                offset += sizeof(cfg_darknet_dropout_layer_info);
                break;
            case NN_CROP:
                break;
            case NN_ROUTE:
                break;
            case NN_COST:
                break;
            case NN_NORMALIZATION:
                break;
            case NN_AVGPOOL:
                break;
            case NN_LOCAL:
                nn_darknet_layer_local_parser((cfg_darknet_local_layer_info*)&((char*)net_info->layers)[offset], object, width, height, channels, batch);
                ((cfg_darknet_local_layer_info*)&((char*)net_info->layers)[offset])->next_layer = (cfg_darknet_layer_info *) (
                        ((char *) net_info->layers) + offset + sizeof(cfg_darknet_local_layer_info));
                height      = ((cfg_darknet_local_layer_info*)&((char*)net_info->layers)[offset])->output_height;
                width       = ((cfg_darknet_local_layer_info*)&((char*)net_info->layers)[offset])->output_width;
                channels    = ((cfg_darknet_local_layer_info*)&((char*)net_info->layers)[offset])->output_channels;
                offset += sizeof(cfg_darknet_local_layer_info);
                break;
            case NN_SHORTCUT:
                break;
            case NN_ACTIVE:
                break;
            case NN_RNN:
                break;
            case NN_GRU:
                break;
            case NN_LSTM:
                break;
            case NN_CRNN:
                break;
            case NN_BATCHNORM:
                break;
            case NN_NETWORK:
                break;
            case NN_XNOR:
                break;
            case NN_REGION:
                nn_darknet_layer_region_parser((cfg_darknet_region_layer_info*)&((char*)net_info->layers)[offset], object, width, height, batch);
                ((cfg_darknet_region_layer_info*)&((char*)net_info->layers)[offset])->next_layer = (cfg_darknet_layer_info *) (
                        ((char *) net_info->layers) + offset + sizeof(cfg_darknet_local_layer_info));
               // height      = ((cfg_darknet_region_layer_info*)&((char*)net_info->layers)[offset])->output_height;
               // width       = ((cfg_darknet_region_layer_info*)&((char*)net_info->layers)[offset])->output_width;
               // channels    = ((cfg_darknet_region_layer_info*)&((char*)net_info->layers)[offset])->channels;
                offset += sizeof(cfg_darknet_region_layer_info) + cfg_option_get_number_values(object, "anchors") * sizeof(float);
                break;
            case NN_YOLO:
                break;
            case NN_ISEG:
                break;
            case NN_REORG:
                break;
            case NN_UPSAMPLE:
                break;
            case NN_LOGXENT:
                break;
            case NN_L2NORM:
                break;
            case NN_BLANK:
                break;
        }
        if (net_info->max_workspace_size < ((struct cfg_darknet_layer_info*)&(((char*)net_info->layers)[last_offset]))->workspace_size) {
            net_info->max_workspace_size = ((struct cfg_darknet_layer_info*)&(((char*)net_info->layers)[last_offset]))->workspace_size;
        }
        object = object->next_type_object;
        last_offset = offset;
    }
}
size_t nn_darknet_layer_get_size_parser(struct cfg_config* options){
    size_t size = 0;
    struct cfg_type_object* object = options->objects;
    for (size_t i = 0; i < options->number_options; i++) {
        switch(get_layer_type(object->current_type_object_value->value)){
            case NN_CONVOLUTIONAL:  size += sizeof(cfg_darknet_convolution_layer_info); break;
            case NN_DECONVOLUTIONAL:  break;
            case NN_CONNECTED: size += sizeof(cfg_darknet_connected_layer_info); break;
            case NN_MAXPOOL: size += sizeof(cfg_darknet_maxpool_layer_info); break;
            case NN_SOFTMAX:  break;
            case NN_DETECTION: size += sizeof(cfg_darknet_detection_layer_info); break;
            case NN_DROPOUT:  size += sizeof(cfg_darknet_dropout_layer_info); break;
            case NN_CROP:  break;
            case NN_ROUTE:  break;
            case NN_COST:   break;
            case NN_NORMALIZATION:  break;
            case NN_AVGPOOL:  break;
            case NN_LOCAL: size += sizeof(cfg_darknet_local_layer_info); break;
            case NN_SHORTCUT:  break;
            case NN_ACTIVE:  break;
            case NN_RNN:  break;
            case NN_GRU:  break;
            case NN_LSTM:   break;
            case NN_CRNN:   break;
            case NN_BATCHNORM:  break;
            case NN_NETWORK:  break;
            case NN_XNOR:   break;
            case NN_REGION: size += sizeof(cfg_darknet_region_layer_info) + cfg_option_get_number_values(object, "anchors") * sizeof(float); break;
            case NN_YOLO:  break;
            case NN_ISEG:  break;
            case NN_REORG:   break;
            case NN_UPSAMPLE:  break;
            case NN_LOGXENT:  break;
            case NN_L2NORM:  break;
            case NN_BLANK:  break;
        }
        object = object->next_type_object;
    }
    return size;
}
cfg_darknet_network_info* nn_darknet_network_parser_malloc(struct cfg_config* options){

    struct cfg_type_object* net_object = nn_darknet_get_net_option_layer(options);

    size_t layers_size = nn_darknet_layer_get_size_parser(options);
    unsigned number_steps = cfg_option_get_number_values(net_object, "steps");
    unsigned number_scales = cfg_option_get_number_values(net_object, "scales");

    cfg_darknet_network_info* net_info = calloc(1, sizeof(cfg_darknet_network_info) + layers_size +
            number_steps * sizeof(int) + number_scales * sizeof(float));
    net_info->number_layers = options->number_options - 1;

    net_info->layers = ((char *) net_info + sizeof(cfg_darknet_network_info));
    net_info->steps = (int *) ((char *) net_info + sizeof(cfg_darknet_network_info) +
            layers_size);
    net_info->scales_or_gamma = (float *) ((char *) net_info + sizeof(cfg_darknet_network_info) +
            layers_size + number_steps * sizeof(int));
    net_info->number_steps =     number_steps;
    net_info->batch =            cfg_option_find_int(net_object, "batch",1);
    net_info->subdivisions =     cfg_option_find_int(net_object, "subdivisions",1);
    net_info->height =           cfg_option_find_int(net_object, "height",0);
    net_info->width =            cfg_option_find_int(net_object, "width",0);
    net_info->channels =         cfg_option_find_int(net_object, "channels",0);
    net_info->max_batches =      cfg_option_find_int(net_object, "max_batches",0);
    net_info->momentum =         cfg_option_find_float(net_object, "momentum", 0.9f);
    net_info->decay =            cfg_option_find_float(net_object, "decay", 0.0001f);
    net_info->saturation =       cfg_option_find_float(net_object, "saturation", 1.0f);
    net_info->exposure =         cfg_option_find_float(net_object, "exposure", 1);
    net_info->hue =              cfg_option_find_float(net_object, "hue", 0);
    net_info->learning_rate =    cfg_option_find_float(net_object, "learning_rate", 0.001f);

    net_info->batch /=           net_info->subdivisions;
    init_policy(net_info, net_object);
    nn_darknet_layer_parser(options, net_info);

    return net_info;
}
void nn_darknet_network_parser_free(cfg_darknet_network_info* cfg_net){
    if (cfg_net) free(cfg_net);
    cfg_net = NULL;
}