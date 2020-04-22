//
// Created by human on 19.03.2020.
//

#include <stddef.h>
#include <float.h>
#include <time.h>
#include <omp.h>
#include "nn_cpu_vectors/longlong2.h"
#include "nn_cpu_vectors/float4.h"
#include "nn_cpu_maxpool_layer.h"

void forward_maxpool_layer(nn_cpu_maxpool_layer* l, nn_cpu_network* net);
void backward_maxpool_layer(nn_cpu_maxpool_layer* l, nn_cpu_network* net);
void nn_cpu_init_maxpool_network(cfg_darknet_maxpool_layer_info* config, nn_cpu_maxpool_layer* layer){
    layer->width = config->width;
    layer->height = config->height;
    layer->channels = config->channels;
    layer->batch = config->batch;
    layer->stride = config->stride;
    layer->filter_length = config->filter_size;
    layer->input_image_length = config->input_image_length;
    layer->padding = config->padding;
    layer->output_width = config->output_width;
    layer->output_height = config->output_height;
    layer->output_channels = config->output_channels;
    layer->output_image_length = config->output_image_length;
    layer->output_batch_length = config->output_batch_length;
    layer->forward = forward_maxpool_layer;
}
void forward_maxpool_layer(nn_cpu_maxpool_layer* l, nn_cpu_network* net)
{
    int h, w, f_h, f_w, i;
    int width = l->width;
    int height = l->height;
    int output_height = l->output_height;
    int output_width = l->output_width;
    int output_batch_channels = l->channels * l->batch;
    int padding = -l->padding;
    int filter_width = l->filter_length;
    int stride = l->stride;
    for(i = omp_get_thread_num(); i < output_batch_channels; i+=omp_get_num_threads()){
        for(h = 0; h < output_height; h++){
            for(w = 0; w < output_width; w++){
                int output_index = w + output_width * (h + output_height * i);
                float maximum = -FLT_MAX;
                int maximum_index = -1;
                for(f_h = 0; f_h < filter_width; ++f_h){
                    for(f_w = 0; f_w < filter_width; ++f_w){
                        int input_height = padding + h*stride + f_h;
                        int input_width = padding + w*stride + f_w;
                        if (input_height >= 0 && input_height < height &&
                                input_width >= 0 && input_width < width){
                            int index = input_width + width*(input_height + height*i);
                            float val = net->input[index];
                            maximum_index = (val > maximum) ? index : maximum_index;
                            maximum   = (val > maximum) ? val   : maximum;
                        }
                    }
                }
                l->output[output_index] = maximum;
                l->indices[output_index] = maximum_index;
            }
        }
    }
    #pragma omp barrier
}
void forward_maxpool_layer_v1(nn_cpu_maxpool_layer* l, nn_cpu_network* net){
    longlong2 filter_xy;
    longlong2 output_image_xy;
    longlong2 width_height = {l->width, l->height};
    longlong2 output_width_height = {l->output_width, l->output_height};
    ptrdiff_t width_filter = l->filter_length;
    ptrdiff_t part_width_filter = l->filter_length - 1;
    ptrdiff_t t, f;
    ptrdiff_t stride = l->stride;
    ptrdiff_t image_hw = l->output_height * l->output_width;
    ptrdiff_t filter_length = l->filter_length * l->filter_length;
    ptrdiff_t tensor_length = l->batch * l->output_image_length;
    ptrdiff_t next_filter_width;
    ptrdiff_t padding = -1ll * l->padding;
    for (t = omp_get_thread_num(); t < tensor_length; t += omp_get_num_threads()){
        float maximum = -FLT_MAX;
        int index_maximum = -1;
        ptrdiff_t index_image = t / image_hw;
        output_image_xy.s[1] = padding + ((t % image_hw) / output_width_height.s[0]) * stride;
        output_image_xy.s[0] = padding + (t % output_width_height.s[0]) * stride;
        filter_xy.s[0] = 0;
        filter_xy.s[1] = 0;
        next_filter_width = part_width_filter;
        for(f = 0; f < filter_length; f++) {
            output_image_xy.s[1] += filter_xy.s[1];
            output_image_xy.s[0] += filter_xy.s[0];

            filter_xy.s[0] = f >= next_filter_width ? -part_width_filter : 1;
            filter_xy.s[1] = f >= next_filter_width ? 1 : 0;
            next_filter_width = f >= next_filter_width ? next_filter_width + width_filter : next_filter_width;

            if ((output_image_xy.s[1] >= 0) & (output_image_xy.s[1] < width_height.s[1]) &
                (output_image_xy.s[0] >= 0) & (output_image_xy.s[0] < width_height.s[0])){
                int index = (index_image * width_height.s[1] + output_image_xy.s[1]) * width_height.s[0] + output_image_xy.s[0];
                float value = net->input[index];
                index_maximum = maximum < value ? index : index_maximum;
                maximum   = maximum < value ? value : maximum;
            }
        }
        l->output[t] = maximum;
        l->indices[t] = index_maximum;
    }
}

void backward_maxpool_layer(nn_cpu_maxpool_layer* l, nn_cpu_network* net)
{

    size_t length = l->output_image_length*l->batch;
    for(size_t i = omp_get_thread_num(); i < length; i+=omp_get_num_threads()){
        size_t index = l->indices[i];
        net->delta[index] += l->delta[i];
    }
#pragma omp barrier
}