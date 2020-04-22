//
// Created by human on 11.03.2020.
//
#include "nn_layers.h"
#include "Parser/cfg_darknet/cfg_reader.h"




/*
char* nn_init_layers(struct cfg_config *options, nn_network *net) {
    size_t height = net->height;
    size_t width = net->width;
    size_t channels = net->channels;
    size_t batch = net->batch;
    size_t inputs = net->inputs;
    size_t time_steps = net->time_steps;
    size_t workspace_size = 0;

    struct cfg_type_object* object = options->objects;
    for (size_t i = 0; i < options->number_options; i++){
        struct cfg_type_object_value* value = object->current_type_object_value;
        //LAYER_TYPE type = get_layer_type(value->value);
        switch (type){
            case NN_CONVOLUTIONAL:
                net->layers[i] = nn_layer_convolutional_parser(object, width, height, channels, batch);
                break;
            case NN_DROPOUT:
                break;
            case NN_DECONVOLUTIONAL:
                break;
            case NN_CONNECTED:
                break;
            case NN_MAXPOOL:
                break;
            case NN_SOFTMAX:
                break;
            case NN_DETECTION:
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
            default:
                break;
        }
        net->layers[i].clip                 = net->clip;
        net->layers[i].truth                = cfg_option_find_int(options, "truth", 0);
        net->layers[i].onlyforward          = cfg_option_find_int(options, "onlyforward", 0);
        net->layers[i].stopbackward         = cfg_option_find_int(options, "stopbackward", 0);
        net->layers[i].dontsave             = cfg_option_find_int(options, "dontsave", 0);
        net->layers[i].dontload             = cfg_option_find_int(options, "dontload", 0);
        net->layers[i].numload              = cfg_option_find_int(options, "numload", 0);
        net->layers[i].dontloadscales       = cfg_option_find_int(options, "dontloadscales", 0);
        net->layers[i].learning_rate_scale  = cfg_option_find_float(options, "learning_rate", 1);
        net->layers[i].smooth               = cfg_option_find_float(options, "smooth", 0);

        if (l.workspace_size > workspace_size) workspace_size = net->layers[i].workspace_size;
        object = object->next;
        if(object){
            height      = net->layers[i].out_h;
            width       = net->layers[i].out_w;
            channels    = net->layers[i].out_c;
            inputs      = net->layers[i].outputs;
        }
    }
}*/