//
// Created by human on 19.03.2020.
//

#ifndef OPENCL_NN_CPU_OPERATIONS_H
#define OPENCL_NN_CPU_OPERATIONS_H
#include <stdlib.h>
#include "nn_cpu_convolution_layer.h"

void nn_cpu_set_float_value_linear(size_t array_size, float* array, float value);
void nn_cpu_set_int_value_linear(size_t array_size, int* array, int value);
void nn_cpu_set_random_float_linear(size_t array_size, float* array, float sigma, float mu);
void nn_cpu_create_convolution_layer_image_ppm(nn_cpu_convolutional_layer* layer, char* file_name, unsigned border_y);
void nn_cpu_create_convolution_layers_images_ppm(nn_cpu_network* layer, unsigned border_y);
void nn_cpu_im2col(float* input, float* output, int offset_input, int offset_output, int channels, int height, int width, int kernel_h, int kernel_w, int stride_x, int stride_y, int padding_x, int padding_y, int height_output, int width_output);
void nn_cpu_im2colv2(float* input, float* output, int offset_input, int offset_output, int channels, int height, int width, int kernel_h, int kernel_w, int stride_x, int stride_y, int padding_x, int padding_y, int height_output, int width_output);
void nn_cpu_col2im(float* input, float* output, int offset_input, int offset_output, int channels, int height, int width, int kernel_h, int kernel_w, int stride_x, int stride_y, int padding_x, int padding_y, int height_output, int width_output);

#endif //OPENCL_NN_CPU_OPERATIONS_H
