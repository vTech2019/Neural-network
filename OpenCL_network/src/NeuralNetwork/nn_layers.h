//
// Created by human on 11.03.2020.
//

#ifndef OPENCL_NN_LAYERS_H
#define OPENCL_NN_LAYERS_H

typedef enum{
    SSE,
    MASKED,
    L1,
    SEG,
    SMOOTH,
    WGAN
} COST_TYPE;

typedef enum {
    NN_LOGISTIC = 1,
    NN_RELU,
    NN_RELIE,
    NN_LINEAR,
    NN_RAMP,
    NN_TANH,
    NN_PLSE,
    NN_LEAKY,
    NN_ELU,
    NN_LOGGY,
    NN_STAIR,
    NN_HARDTAN,
    NN_LHTAN,
    NN_SELU
} NN_ACTIVATION_FUNCTION;

typedef enum {
    NN_CONVOLUTIONAL = 1,
    NN_DECONVOLUTIONAL,
    NN_CONNECTED,
    NN_MAXPOOL,
    NN_SOFTMAX,
    NN_DETECTION,
    NN_DROPOUT,
    NN_CROP,
    NN_ROUTE,
    NN_COST,
    NN_NORMALIZATION,
    NN_AVGPOOL,
    NN_LOCAL,
    NN_SHORTCUT,
    NN_ACTIVE,
    NN_RNN,
    NN_GRU,
    NN_LSTM,
    NN_CRNN,
    NN_BATCHNORM,
    NN_NETWORK,
    NN_XNOR,
    NN_REGION,
    NN_YOLO,
    NN_ISEG,
    NN_REORG,
    NN_UPSAMPLE,
    NN_LOGXENT,
    NN_L2NORM,
    NN_BLANK
} NN_LAYER_TYPE;

//void nn_l_init_layer(struct nn_layer* init_layer);
//void free_layer(struct nn_layer* layer);

#endif //OPENCL_NN_LAYERS_H
