//
// Created by human on 19.03.2020.
//

#include <ppm_reader.h>
#include <lzma.h>
#include <omp.h>
#include "nn_cpu_operations.h"
#include "../../random/random.h"

void itoa(const char* start, char* end, int val, int base){
    int i = 30;
    end--;
    for(; start < end && i ; --i, val /= base, end--)
        *end = "0123456789abcdef"[val % base];
}
void nn_cpu_set_int_value_linear(size_t array_size, int* array, int value)
{
    for(int* end = array + array_size; array < end; array++) *array = value;
}
void nn_cpu_set_float_value_linear(size_t array_size, float* array, float value)
{
    for(float* end = array + array_size; array < end; array++) *array = value;
}
void nn_cpu_set_random_float_linear(size_t array_size, float* array, float sigma, float mu)
{
    size_t i;
    for(i = 0; i + 1 < array_size; i += 2) random_GaussianNoiseBoxMuller(&array[i], &array[i + 1], mu, sigma);
    if (i != array_size) random_GaussianNoiseBoxMuller(&array[i], &array[i], mu, sigma);
}

void nn_cpu_normalize_image(float* image, unsigned char* result, size_t width_result, size_t width, size_t height, size_t channels)
{
    size_t length = width * height * channels;
    float min = FLT_MAX;
    float max = FLT_MIN;
    for (size_t i = 0; i < length; i++){
        float value = image[i];
        min = value > min ? min : value;
        max = value < max ? max : value;
    }
    if(max - min < FLT_EPSILON){
        min = 0;
        max = 1;
    }
    float border = 255.0f / (max - min);
    for(size_t c = 0, k = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++, k++) {
                result[c + h * width_result + w * channels] = border * (image[k] - min);
                //printf("%f ", border * (image[k] - min));
            }
        }
        //printf("\n");
    }
}

void nn_cpu_create_convolution_layer_image_ppm(nn_cpu_convolutional_layer* layer, char* file_name, unsigned border_y)
{
    float* ptr_layer = layer->weight;
    unsigned n_height = layer->filters;
    unsigned border_length_h = n_height == 1 ? 0 : border_y;

    unsigned n_height_length = (layer->filter_length + border_length_h) * (layer->filter_length) * layer->channels;
    unsigned n_l_height_length = layer->filter_length * layer->filter_length * layer->channels;

    unsigned width_length = layer->channels * (layer->filter_length);

    size_t size_image = n_height * (layer->filter_length + border_length_h) * (layer->filter_length) * layer->channels;
    unsigned char* image = calloc(1, size_image);
    for (unsigned n_h = 0; n_h < n_height; n_h++){
        unsigned char* ptr_n_h = image + n_h * n_height_length;
        float* ptr_l_n_h = ptr_layer + n_h * n_l_height_length;
        nn_cpu_normalize_image(ptr_l_n_h, ptr_n_h, width_length, layer->filter_length, layer->filter_length, layer->channels);
    }
    if (layer->channels == 3)
        create_ppm_image(file_name, image, (layer->filter_length), n_height * (layer->filter_length + border_length_h), 255);
    else
        create_pgm_image(file_name, image, (layer->filter_length) * layer->channels, n_height * (layer->filter_length + border_length_h), 255);
    if (image) free(image);
}
void nn_cpu_create_convolution_layers_images_ppm(nn_cpu_network* layer, unsigned border_y)
{
    ptrdiff_t i = 0, thread_data = 0;
    char layer_name[] = "layer_000.ppm";
    nn_cpu_layer* start_layer = layer->layers;
    #pragma omp parallel for firstprivate(thread_data, start_layer, layer_name) private(i)
    for (i = 0; i < layer->number_layers; i++){
        for (size_t j = thread_data; j < i; j++){
            start_layer = start_layer->next_layer;
            thread_data = j;
        }
        itoa(layer_name+5, layer_name+9, i, 10);
        if (start_layer->type_layer == NN_CONVOLUTIONAL){
            nn_cpu_create_convolution_layer_image_ppm(start_layer, layer_name, border_y);
        }
    }
}

void nn_cpu_im2col(float* input, float* output, int offset_input, int offset_output,
                   int channels, int height, int width,
                   int kernel_h, int kernel_w,
                   int stride_x, int stride_y,
                   int padding_x, int padding_y,
                   int height_output, int width_output)
{
    int c, c_out, c_inp;
    int h, h_k, h_out, h_inp;
    int w, w_k, w_out, w_inp;
    int id_thread = omp_get_thread_num();
    int number_thread = omp_get_num_threads();
    float* ptr_input;
    float* ptr_output;
    for (c = 0; c < channels; ++c) {
        c_inp = c;
        c_out = c * kernel_h * kernel_w;
        for (h = id_thread; h < height_output; h+=number_thread) {
            h_inp = h * stride_y - padding_y;
            h_out = h;
            for (w = 0; w < width_output; ++w) {
                w_inp = w * stride_x - padding_x;
                w_out = w;
                ptr_input = input + (c_inp * height + h_inp) * width + w_inp + offset_input;
                ptr_output = output + (c_out * height_output + h_out) * width_output + w_out + offset_output;
                for (h_k = 0; h_k < kernel_h; h_k++)
                    for (w_k = 0; w_k < kernel_w; w_k++) {
                        register unsigned i = h_inp + h_k;
                        register unsigned j = w_inp + w_k;
                        *ptr_output = (i < height && j < width) ? ptr_input[h_k * width + w_k] : 0;
                        ptr_output += height_output * width_output;
                    }
            }
        }
    }
}

void nn_cpu_col2im(float* input, float* output, int offset_input, int offset_output,
                   int channels, int height, int width,
                   int kernel_h, int kernel_w,
                   int stride_x, int stride_y,
                   int padding_x, int padding_y,
                   int height_output, int width_output)
{
    int c, c_out, c_inp;
    int h, h_out, h_inp;
    int w, w_out, w_inp;
    int id_thread = omp_get_thread_num();
    int number_thread = omp_get_num_threads();
    float* ptr_input;
    for (c = 0; c < channels; ++c) {
        c_inp = c * kernel_h * kernel_w;
        c_out = c;
        for (h = id_thread; h < height_output; h+=number_thread) {
            h_out = h * stride_y - padding_y;
            h_inp = h;
            for (w = 0; w < width_output; ++w) {
                w_out = w * stride_x - padding_x;
                w_inp = w;
                ptr_input = input + (c_inp * height + h_inp) * width + w_inp + offset_input;
                if (h_out >= 0 && h_out < height && w_out >= 0 && w_out < width){
                    output[(c_out * height_output + h_out) * width_output + w_out + offset_output] = *ptr_input;
                }
            }
        }
    }
}

void nn_cpu_im2colv2(float* input, float* output, int offset_input, int offset_output,
                   int channels, int height, int width,
                   int kernel_h, int kernel_w,
                   int stride_x, int stride_y,
                   int padding_x, int padding_y,
                   int height_output, int width_output)
{
    int c, c_out, c_inp;
    int h, h_k, h_out, h_inp;
    int w, w_k, w_out, w_inp;
    int id_thread = omp_get_thread_num();
    int number_thread = omp_get_num_threads();
    float* ptr_input;
    float* ptr_output;
    for (c = 0; c < channels; ++c) {
        c_inp = c * height * width;
        c_out = c * height_output * width_output;
        for (h = id_thread; h < kernel_h; h+=number_thread) {
            h_inp = h * stride_y - padding_y;
            h_out = h * width_output;
            for (w = 0; w < kernel_w; ++w) {
                printf("\n");
                w_inp = w * stride_x - padding_x;
                w_out = w;
                ptr_input = input + c_inp + h_inp * width + w_inp + offset_input;
                ptr_output = output + (c_out + h_out + w_out)*(width_output * height_output) + offset_output;
                for (h_k = 0; h_k < height_output; h_k++)
                    for (w_k = 0; w_k < width_output; w_k++) {
                        register unsigned i = h_inp + h_k;
                        register unsigned j = w_inp + w_k;
                        *ptr_output = (i < height && j < width) ? ptr_input[h_k * width + w_k] : 0;
                        printf("%f ", *ptr_output );
                        ptr_output++;
                        fflush (stdout);
                    }
            }
        }
    }
}