//
// Created by human on 11.03.2020.
//
#include <omp.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include "nn_cpu_convolution_layer.h"
#include "nn_cpu_activation.h"
#include "nn_cpu_operations.h"
#include "nn_cpu_vectors/float8.h"

void forward_convolutional_layer(nn_cpu_convolutional_layer* l, nn_cpu_network* net);
void backward_convolutional_layer(nn_cpu_convolutional_layer* l, nn_cpu_network* net);
void update_convolutional_layer(nn_cpu_convolutional_layer* l);

void nn_cpu_init_convolution_network(cfg_darknet_convolution_layer_info* config, nn_cpu_convolutional_layer* layer){
    layer->batch = config->batch;
    layer->width = config->width;
    layer->height = config->height;
    layer->channels = config->channels;
    layer->input_length = config->width * config->height * config->channels;
    layer->output_width = config->output_width;
    layer->output_height = config->output_height;
    layer->output_channels = config->output_channels;
    layer->output_length = config->output_width * config->output_height * config->output_channels;
    layer->filters = config->filters;
    layer->filters_weight_length = config->weight_length;
    layer->filter_length = config->filter_length;
    layer->stride = config->stride;
    layer->padding = config->padding;
    layer->groups = config->groups;
    layer->batch_normalize = config->batch_normalize;
    layer->transposed = config->transposed;
    layer->workspace_size = config->workspace_size;
    layer->function = config->function;
    layer->forward = forward_convolutional_layer;
    layer->backward = backward_convolutional_layer;
    layer->update = update_convolutional_layer;
}

float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}
void col2im_add_pixel(float *im, int height, int width, int channels,
                      int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}
//This one might be too, can't remember.
void col2im_cpu(float* data_col,
                int channels,  int height,  int width,
                int ksize,  int stride, int pad, float* data_im)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                                 im_row, im_col, c_im, pad, val);
            }
        }
    }
}

float im2col_get_pixel(float *im, int height, int width, int channels,
                       int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

void im2col_cpu(float* data_im,
                int channels,  int height,  int width,
                int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad);
                //if (data_col[col_index])
                //printf("%f\n", data_col[col_index]);
            }
        }
    }
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}


void gemm_01(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float BETA,
             float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
#pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_10(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float BETA,
             float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
#pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void tensor_convolution(float* data, float* result, int batch, int channels, int height, int width, float* kernel, int filters, int width_k, int height_k){
    for (int i = 0; i < batch; i++){
        for (int h = 0; h < height; h++){
            for (int w = 0; w < width; w++){
                int index = (((i * channels)) * height + h) * width + w;
                for (int k_f = 0; k_f < filters; k_f++) {
                    float sum = 0;
                    for (int k_c = 0; k_c < channels; k_c++) {
                        for (int k_h = 0; k_h < height_k; k_h++) {
                            if ((h + k_h) >= 0 && (h + k_h) < height)
                                for (int k_w = 0; k_w < width; k_w++) {
                                    if ((w + k_w) >= 0 && (w + k_w) < width)
                                        sum += kernel[((k_f * channels + k_c) * height_k + k_h) * width_k + k_w] *
                                                   data[index + (k_c * height + k_h) * width + k_w];
                                }
                        }
                    }
                    if (sum != result[k_f * width * height + index])
                        printf("Error");
                    result[k_f * width * height + index] = sum;
                }
            }
        }
    }
}

void forward_convolutional_layer(nn_cpu_convolutional_layer* l, nn_cpu_network* net)
{
    int i, end, f_n, f_s, i_j;
    float *image = net->workspace;
    float *output;
    unsigned thread_id = omp_get_thread_num();
    unsigned thread_number = omp_get_num_threads();
    unsigned filters = l->filters;
    unsigned filter_length = l->filter_length*l->filter_length*l->channels;
    unsigned output_hw = l->output_width*l->output_height;
    unsigned step_image = l->channels * l->height * l->width;
#pragma omp single
    {for(i = 0, end = l->output_channels * l->output_width * l->output_height * l->batch; i < end; ++i) l->output[i] = 0;}
#pragma omp single
    {for(i = 0, end = l->output_channels * l->output_width * l->output_height * l->batch; i < end; ++i) l->delta[i] = 0;}

#pragma omp barrier
    for(i = 0; i < l->batch; i++){
        output = l->output + i * output_hw * filters;
        nn_cpu_im2col(net->input + i * step_image, image, 0, 0,
                        l->channels, l->height, l->width,
                        l->filter_length, l->filter_length,
                        l->stride, l->stride,
                        l->padding, l->padding,
                        l->output_height, l->output_width);
        for(f_n = 0; f_n < filters; ++f_n){
            for(f_s = thread_id; f_s < filter_length; f_s+=thread_number){
                register float value = l->weight[f_n * filter_length + f_s];
                for(i_j = 0; i_j < output_hw; i_j++){
                    output[f_n * output_hw + i_j] += value * image[f_s * output_hw + i_j];
                }
            }
        }
    }
    for(i = 0, end = l->output_height*l->output_width; i < l->batch; i++){
        output = l->output + (i*filters)*output_hw;
        for(f_n = thread_id; f_n < filters; f_n+=thread_number){
            register float bias = l->bias[f_n];
            for(f_s = 0; f_s < output_hw; ++f_s){
                l->output[(i*filters + f_n)*output_hw + f_s] += bias;
            }
        }
    }
#pragma omp single
{
    for (i = 0, end = l->output_width*l->output_height*l->output_channels*l->batch; i < end; i++)
        l->output[i] = nn_cpu_get_activation_function(l->function, l->output[i]);
};
#pragma omp barrier
}

void backward_convolutional_layer_v1(nn_cpu_convolutional_layer* l, nn_cpu_network* net)
{
    int i, j;
    int m = l->filters/l->groups;
    int k = l->filter_length*l->filter_length*l->channels/l->groups;
    int n = l->output_width*l->output_height;

    nn_cpu_gradient(l->output, l->delta, l->output_width*l->output_height*l->channels*l->batch, l->function);
    backward_bias(l->bias_update, l->delta, l->batch, l->filters, k);

    for(i = 0; i < l->batch; ++i){
        for(j = 0; j < l->groups; ++j){
            float *a = l->delta + (i*l->groups + j)*m*k;
            float *b = net->workspace;
            float *c = l->weight_update + j * l->filters_weight_length/l->groups;

            float *im  = net->input + (i*l->groups + j)*l->channels/l->groups*l->height*l->width;
            float *imd = net->delta + (i*l->groups + j)*l->channels/l->groups*l->height*l->width;

            if(l->filter_length == 1){
                b = im;
            } else {
                im2col_cpu(im, l->channels/l->groups, l->height, l->width, l->filter_length, l->stride, l->padding, b);
            }

            gemm_01(m,n,k,1,a,k,b,k,1,c,n);

            if (net->delta) {
                a = l->weight + j*l->filters_weight_length/l->groups;
                b = l->delta + (i*l->groups + j)*m*k;
                c = net->workspace;
                if (l->filter_length == 1) {
                    c = imd;
                }

                gemm_10(n,k,m,1,a,n,b,k,0,c,k);

                if (l->filter_length != 1) {
                    col2im_cpu(net->workspace, l->channels/l->groups, l->height, l->width, l->filter_length, l->stride, l->padding, imd);
                }
            }
        }
    }
}
void backward_convolutional_layer(nn_cpu_convolutional_layer* l, nn_cpu_network* net)
{
    int i, j, end;
    int m = l->filters;
    int k = l->filter_length*l->filter_length*l->channels;
    int n = l->output_width*l->output_height;

#pragma omp single
    {
        for (i = 0, end = l->output_width*l->output_height*l->channels*l->batch; i < end; i++)
            l->output[i] = nn_cpu_get_gradient_function(l->function, l->output[i]);
    };
    for(i = 0; i < l->batch; i++){
        for(j = 0; j < n; j++){
            for(i = 0; i < k; ++i)
                l->bias_update[i] += l->delta[( j + i * n ) * k + i];
        }
    }
    for(i = 0; i < l->batch; ++i) {
        float *a = l->delta + i*m*k;
        float *b = net->workspace;
        float *c = l->weight_update;
        float *im = net->input+i*l->channels*l->height*l->width;

        nn_cpu_im2col(im, b, 0, 0,
                      l->channels, l->height, l->width,
                      l->filter_length, l->filter_length,
                      l->stride, l->stride,
                      l->padding, l->padding,
                      l->output_height, l->output_width);

#pragma omp parallel for
            for(i = 0; i < m; ++i){
                for(j = 0; j < n; ++j){
                    register float sum = 0;
                    for(k = 0; k < k; ++k){
                        sum += a[i*k+k]*b[j*k + k];
                    }
                    c[i*n+j] += sum;
                }
            }

        if(net->delta){
            a = l->weight;
            b = l->delta + i*m*k;
            c = net->workspace;

                for(i = 0; i < n; ++i){
                    for(j = 0; j < k; ++j){
                        c[i*k + j] = 0;
                    }
                }
                for(i = 0; i < n; ++i){
                    for(k = 0; k < m; ++k){
                        register float A_PART = a[k*n+i];
                        for(j = 0; j < k; ++j){
                            c[i*k+j] += A_PART*b[k*k+j];
                        }
                    }
                }
            }
            col2im_cpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta+i*l.c*l.h*l.w);
        }

    }
}

void update_convolutional_layer(nn_cpu_convolutional_layer* l)
{
    /*float learning_rate = a.learning_rate * l->learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);*/
}

