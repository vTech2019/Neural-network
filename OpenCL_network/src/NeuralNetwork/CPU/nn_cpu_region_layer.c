//
// Created by human on 19.03.2020.
//
#include <string.h>
#include <float.h>

#include "nn_cpu_region_layer.h"
#include "nn_cpu_activation.h"

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;
    int groups;
    int *group_size;
    int *group_offset;
} tree;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

void forward_region_layer(nn_cpu_region_layer* l, nn_cpu_network* net);
void backward_region_layer(const nn_cpu_region_layer* l, nn_cpu_network* net);

void nn_cpu_init_region_network(cfg_darknet_region_layer_info* config, nn_cpu_region_layer* layer){
    layer->classes = config->classes;
    layer->coords = config->coords;
    layer->num = config->num;
    layer->width = config->width;
    layer->height = config->height;
    layer->channels = config->channels;
    layer->batch = config->batch;
    layer->softmax = config->softmax;
    //layer->softmax_tree = config->softmax_tree;
    layer->output_height = config->output_height;
    layer->output_width = config->output_width;
    layer->output_channels = config->output_channels;
    layer->number_biases = config->number_biases;
    layer->output_image_length = config->output_height * config->output_width * config->output_channels;
    layer->input_image_length = config->height * config->width * config->channels;
    layer->background = config->background;
    layer->truths = config->truths;
    layer->noobject_scale = config->noobject_scale;
    layer->thresh = config->thresh;
    layer->bias_match = config->bias_match;
    layer->object_scale = config->object_scale;
    layer->rescore = config->rescore;
    layer->coord_scale = config->coord_scale;
    layer->class_scale = config->class_scale;
    layer->mask_scale = config->mask_scale;
    layer->softmax_tree = NULL;
    for (size_t i = 0; i < config->number_biases; i++)
        layer->biases[i] = config->biases[i];
    layer->cost = 0;

    layer->forward = forward_region_layer;
    layer->backward = backward_region_layer;
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i*stride] > largest) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

int entry_index(nn_cpu_region_layer* l, int batch, int location, int entry)
{
    int n =   location / (l->width*l->height);
    int loc = location % (l->width*l->height);
    return batch*l->output_image_length + n*l->width*l->height*(l->coords+l->classes+1) + entry*l->width*l->height + loc;
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}
float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

box float_to_box(float *f, int stride)
{
    box b = {0};
    b.x = f[0];
    b.y = f[1*stride];
    b.w = f[2*stride];
    b.h = f[3*stride];
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

float mag_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        sum += a[i]*a[i];
    }
    return sqrt(sum);
}

void delta_region_class(float *output, float *delta, int index, int class, int classes, tree *hier, float scale, int stride, float *avg_cat, int tag)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class >= 0){
            pred *= output[index + stride*class];
            int g = hier->group[class];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);

            class = hier->parent[class];
        }
        *avg_cat += pred;
    } else {
        if (delta[index] && tag){
            delta[index + stride*class] = scale * (1 - output[index + stride*class]);
            return;
        }
        for(n = 0; n < classes; ++n){
            delta[index + stride*n] = scale * (((n == class)?1 : 0) - output[index + stride*n]);
            if(n == class) *avg_cat += output[index + stride*n];
        }
    }
}

void delta_region_mask(float *truth, float *x, int n, int index, float *delta, int stride, int scale)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[index + i*stride] = scale*(truth[i] - x[index + i*stride]);
    }
}

void forward_region_layer(nn_cpu_region_layer* l, nn_cpu_network* net)
{
    int i,j,t;
    memcpy(l->output, net->input, l->output_image_length*l->batch*sizeof(float));

    for (unsigned b = 0; b < l->batch; ++b){
        for(unsigned n = 0; n < l->num; ++n){
            unsigned location = n * l->width * l->height;
            unsigned loc = location % (l->width * l->height);
            unsigned index = b * l->output_image_length + n * l->width * l->height * (l->coords + l->classes+1) + loc;
            nn_cpu_activate(l->output + index, 2 * l->width * l->height, NN_LOGISTIC);
            index = b * l->output_image_length + n * l->width * l->height * (l->coords + l->classes+1) + l->coords * l->width * l->height + loc;
            if(!l->background) nn_cpu_activate(l->output + index, l->width * l->height, NN_LOGISTIC);
            index = b * l->output_image_length + n * l->width * l->height * (l->coords + l->classes+1) + (l->coords + 1) * l->width * l->height + loc;
            //if(!l->softmax && !l->softmax_tree)
                nn_cpu_activate(l->output + index, l->classes * l->width *l->height, NN_LOGISTIC);
        }
    }
    /*if (l.softmax_tree){
        int i;
        int count = l.coords + 1;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
            count += group_size;
        }
    } else */if (l->softmax) {
        unsigned index = l->width * l->height * (l->coords + !l->background);
        softmax_cpu(net->input + index, l->classes + l->background, l->batch * l->num, l->input_image_length / l->num,
                    l->width * l->height, 1, l->width * l->height, 1, l->output + index);
    }

    memset(l->delta, 0, l->output_image_length * l->batch * sizeof(float));
    if(!net->train) return;

    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    (l->cost) = 0;
    for (unsigned b = 0; b < l->batch; ++b) {
        /*if(l.softmax_tree){
            int onlyclass = 0;
            for(t = 0; t < 30; ++t){
                box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
                if(!truth.x) break;
                int class = net.truth[t*(l.coords + 1) + b*l.truths + l.coords];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int class_index = entry_index(l, b, n, l.coords + 1);
                        int obj_index = entry_index(l, b, n, l.coords);
                        float scale =  l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);
                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, class, l.w*l.h);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index(l, b, maxi, l.coords + 1);
                    int obj_index = entry_index(l, b, maxi, l.coords);
                    delta_region_class(l.output, l.delta, class_index, class, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, &avg_cat, !l.softmax);
                    if(l.output[obj_index] < .3) l.delta[obj_index] = l.object_scale * (.3 - l.output[obj_index]);
                    else  l.delta[obj_index] = 0;
                    l.delta[obj_index] = 0;
                    ++class_count;
                    onlyclass = 1;
                    break;
                }
            }
            if(onlyclass) continue;
        }*/
        for (unsigned j = 0; j < l->height; ++j) {
            for (unsigned i = 0; i < l->width; ++i) {
                for (unsigned n = 0; n < l->num; ++n) {
                    int box_index = entry_index(l, b, n*l->width*l->height + j*l->width + i, 0);
                    box pred = get_region_box(l->output, l->biases, n, box_index, i, j, l->width, l->height, l->width*l->height);
                    float best_iou = 0;
                    for(t = 0; t < 30; ++t){
                        box truth = float_to_box(net->truth + t*(l->coords + 1) + b*l->truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                        }
                    }
                    int obj_index = entry_index(l, b, n*l->width*l->height + j*l->width + i, l->coords);
                    avg_anyobj += l->output[obj_index];
                    l->delta[obj_index] = l->noobject_scale * (0 - l->output[obj_index]);
                    if(l->background) l->delta[obj_index] = l->noobject_scale * (1 - l->output[obj_index]);
                    if (best_iou > l->thresh) {
                        l->delta[obj_index] = 0;
                    }

                    if(net->seen < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l->width;
                        truth.y = (j + .5)/l->height;
                        truth.w = l->biases[2*n]/l->width;
                        truth.h = l->biases[2*n+1]/l->height;
                        delta_region_box(truth, l->output, l->biases, n, box_index, i, j, l->width, l->height, l->delta, .01, l->width*l->height);
                    }
                }
            }
        }
        for(t = 0; t < 30; ++t){
            box truth = float_to_box(net->truth + t*(l->coords + 1) + b*l->truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l->width);
            j = (truth.y * l->height);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            for(unsigned n = 0; n < l->num; ++n){
                int box_index = entry_index(l, b, n*l->width*l->height + j*l->width + i, 0);
                box pred = get_region_box(l->output, l->biases, n, box_index, i, j, l->width, l->height, l->width*l->height);
                if(l->bias_match){
                    pred.w = l->biases[2*n]/l->width;
                    pred.h = l->biases[2*n+1]/l->height;
                }
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }

            int box_index = entry_index(l, b, best_n*l->width*l->height + j*l->width + i, 0);
            float iou = delta_region_box(truth, l->output, l->biases, best_n, box_index, i, j, l->width, l->height, l->delta, l->coord_scale *  (2 - truth.w*truth.h), l->width*l->height);
            if(l->coords > 4){
                int mask_index = entry_index(l, b, best_n*l->width*l->height + j*l->width + i, 4);
                delta_region_mask(net->truth + t*(l->coords + 1) + b*l->truths + 5, l->output, l->coords - 4, mask_index, l->delta, l->width*l->height, l->mask_scale);
            }
            if(iou > .5) recall += 1;
            avg_iou += iou;

            int obj_index = entry_index(l, b, best_n*l->width*l->height + j*l->width + i, l->coords);
            avg_obj += l->output[obj_index];
            l->delta[obj_index] = l->object_scale * (1 - l->output[obj_index]);
            if (l->rescore) {
                l->delta[obj_index] = l->object_scale * (iou - l->output[obj_index]);
            }
            if(l->background){
                l->delta[obj_index] = l->object_scale * (0 - l->output[obj_index]);
            }

            int class = net->truth[t*(l->coords + 1) + b*l->truths + l->coords];
           // if (l->map) class = l->map[class];
            int class_index = entry_index(l, b, best_n*l->width*l->height + j*l->width + i, l->coords + 1);
            delta_region_class(l->output, l->delta, class_index, class, l->classes, l->softmax_tree, l->class_scale, l->width*l->height, &avg_cat, !l->softmax);
            ++count;
            ++class_count;
        }
    }
    l->cost = pow(mag_array(l->delta, l->output_image_length * l->batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l->width*l->height*l->num*l->batch), recall/count, count);

}

void backward_region_layer(const nn_cpu_region_layer* l, nn_cpu_network* net)
{
    /*
       int b;
       int size = l.coords + l.classes + 1;
       for (b = 0; b < l.batch*l.n; ++b){
       int index = (b*size + 4)*l.w*l.h;
       gradient_array(l.output + index, l.w*l.h, LOGISTIC, l.delta + index);
       }
       axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
     */
}
void update_region_layer(nn_cpu_region_layer* l) {


}

void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride)
{
    int j;
    for(j = 0; j < n; ++j){
        int parent = hier->parent[j];
        if(parent >= 0){
            predictions[j*stride] *= predictions[parent*stride];
        }
    }
    if(only_leaves){
        for(j = 0; j < n; ++j){
            if(!hier->leaf[j]) predictions[j*stride] = 0;
        }
    }
}

int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride)
{
    float p = 1;
    int group = 0;
    int i;
    while(1){
        float max = 0;
        int max_i = 0;

        for(i = 0; i < hier->group_size[group]; ++i){
            int index = i + hier->group_offset[group];
            float val = predictions[(i + hier->group_offset[group])*stride];
            if(val > max){
                max_i = index;
                max = val;
            }
        }
        if(p*max > thresh){
            p = p*max;
            group = hier->child[max_i];
            if(hier->child[max_i] < 0) return max_i;
        } else if (group == 0){
            return max_i;
        } else {
            return hier->parent[hier->group_offset[group]];
        }
    }
    return 0;
}

void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

void get_region_detections(nn_cpu_region_layer* l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i,j,n,z;
    float *predictions = l->output;
    if (l->batch == 2) {
        float *flip = l->output + l->output_image_length;
        for (j = 0; j < l->height; ++j) {
            for (i = 0; i < l->width/2; ++i) {
                for (n = 0; n < l->num; ++n) {
                    for(z = 0; z < l->classes + l->coords + 1; ++z){
                        int i1 = z*l->width*l->height*l->num + n*l->width*l->height + j*l->width + i;
                        int i2 = z*l->width*l->height*l->num + n*l->width*l->height + j*l->width + (l->width - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if(z == 0){
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for(i = 0; i < l->output_image_length; ++i){
            l->output[i] = (l->output[i] + flip[i])/2.;
        }
    }
    for (i = 0; i < l->width*l->height; ++i){
        int row = i / l->width;
        int col = i % l->width;
        for(n = 0; n < l->num; ++n){
            int index = n*l->width*l->height + i;
            for(j = 0; j < l->classes; ++j){
                dets[index].prob[j] = 0;
            }
            int obj_index  = entry_index(l, 0, n*l->width*l->height + i, l->coords);
            int box_index  = entry_index(l, 0, n*l->width*l->height + i, 0);
            int mask_index = entry_index(l, 0, n*l->width*l->height + i, 4);
            float scale = l->background ? 1 : predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l->biases, n, box_index, col, row, l->width, l->height, l->width*l->height);
            dets[index].objectness = scale > thresh ? scale : 0;
            if(dets[index].mask){
                for(j = 0; j < l->coords - 4; ++j){
                    dets[index].mask[j] = l->output[mask_index + j*l->width*l->height];
                }
            }

            int class_index = entry_index(l, 0, n*l->width*l->height + i, l->coords + !l->background);
            if(l->softmax_tree){

                hierarchy_predictions(predictions + class_index, l->classes, l->softmax_tree, 0, l->width*l->height);
                if(map){
                    for(j = 0; j < 200; ++j){
                        int class_index = entry_index(l, 0, n*l->width*l->height + i, l->coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    int j =  hierarchy_top_prediction(predictions + class_index, l->softmax_tree, tree_thresh, l->width*l->height);
                    dets[index].prob[j] = (scale > thresh) ? scale : 0;
                }
            } else {
                if(dets[index].objectness){
                    for(j = 0; j < l->classes; ++j){
                        int class_index = entry_index(l, 0, n*l->width*l->height + i, l->coords + 1 + j);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
            }
        }
    }
    correct_region_boxes(dets, l->width*l->height*l->num, w, h, netw, neth, relative);
}