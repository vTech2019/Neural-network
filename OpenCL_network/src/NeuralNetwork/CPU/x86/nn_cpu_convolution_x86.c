//
// Created by human on 01.04.2020.
//

#include <omp.h>
#include "nn_cpu_convolution_x86.h"
#include <immintrin.h>

void forward_convolutional_layer_x86(nn_cpu_convolutional_layer* l, nn_cpu_network* net){

    int i, j, h, w, end;
    float *image = net->workspace;
    float *output = l->output;
    int thread_id = omp_get_thread_num();
    int thread_number = omp_get_num_threads();
    int filters = l->filters;
    int filter_length = l->filter_length*l->filter_length*l->channels;
    int step_image = l->channels * l->height * l->width;
    int batch = l->batch;
    int output_hw = l->output_width*l->output_height;
    int output_length = l->output_length;
    for (i = thread_id, end = output_length / 8; i < batch; i+=thread_number){
        __m256* vector = (__m256 *) (l->output + i * output_length);
        __m256 vector_zero = _mm256_set1_ps(0);
        for (j = 0; j < end; j++)
            _mm256_store_ps(vector + j, vector_zero);
        for (j *= 8; j < output_length; j++)
            l->output[j] = 0;
    }
}