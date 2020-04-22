//
// Created by human on 05.03.2020.
//
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>
#include <limits.h>
#include "../../File/file.h"
#include "cfg_reader.h"

void node_error(char *s){
    fprintf(stderr, "Node error: %s\n", s);
    exit(0);
}

char* cfg_option_find(struct cfg_type_object* config, char *option)
{
        for (struct cfg_type_object_value* iter = config->current_type_object_value; iter; iter = iter->next){
            switch (iter->type) {
                case CFG_OBJECT:
                    break;
                case CFG_TYPE_NAME:
                    if(strcmp(iter->value, option) == 0){
                        iter->next->type = CFG_TYPE_USED;
                        return iter->next->value;
                    }
                    break;
                case CFG_TYPE_VALUE:
                    break;
                case CFG_TYPE_USED:
                    break;
            }
        }
    return NULL;
}

int cfg_option_find_int(struct cfg_type_object* config, char *option, int default_value)
{
    char *text = cfg_option_find(config, option);
    if(text) return atoi(text);
    return default_value;
}
float cfg_option_find_float(struct cfg_type_object* config, char *option, float default_value)
{
    char *text = cfg_option_find(config, option);
    if(text) return atof(text);
    return default_value;
}
char* cfg_option_find_string(struct cfg_type_object* config, char *option, char* default_value)
{
    char *text = cfg_option_find(config, option);
    if(text) return text;
    return default_value;
}

unsigned cfg_option_find_floats_return_number_write(struct cfg_type_object* config, char *option, float return_values[], unsigned number_values)
{
    char *text = cfg_option_find(config, option);
    if(!text) return 0;
    unsigned int length_steps = strlen(text);
    unsigned l_number_values = 1;
    for(unsigned i = 0; i < length_steps; ++i)
        if (text[i] == ',') l_number_values++;
    l_number_values = l_number_values < number_values ? l_number_values : number_values;
    for(unsigned int i = 0; i < l_number_values; ++i){
        float step = atof(text);
        text = strchr(text, ',') + 1;
        return_values[i] = step;
    }
    return l_number_values;
}
unsigned cfg_option_find_ints_return_number_write(struct cfg_type_object* config, char *option, int return_values[], unsigned number_values)
{
    char *text = cfg_option_find(config, option);
    if(!text) return 0;
    unsigned int length_steps = strlen(text);
    unsigned l_number_values = 1;
    for(unsigned i = 0; i < length_steps; ++i)
        if (text[i] == ',') l_number_values++;
    l_number_values = l_number_values < number_values ? l_number_values : number_values;
    for(unsigned int i = 0; i < l_number_values; ++i){
        int step = atoi(text);
        text = strchr(text, ',') + 1;
        return_values[i] = step;
    }
    return l_number_values;
}

float* cfg_option_find_floats_malloc(struct cfg_type_object* config, char *option, int* number_values)
{
    char *text = cfg_option_find(config, option);
    if(!text) return NULL;
    unsigned length_steps = strlen(text);
    unsigned l_number_values = 1;
    for(unsigned int i = 0; i < length_steps; ++i)
        if (text[i] == ',') l_number_values++;
    float *steps = calloc(l_number_values, sizeof(float));
    for(unsigned i = 0; i < l_number_values; ++i){
        float step = atof(text);
        text = strchr(text, ',') + 1;
        steps[i] = step;
    }
    if (number_values)
        *number_values = l_number_values;
    return steps;
}

int* cfg_option_find_ints_malloc(struct cfg_type_object* config, char *option, int* number_values)
{
    char *text = cfg_option_find(config, option);
    if(!text) return NULL;
    unsigned length_steps = strlen(text);
    unsigned l_number_values = 1;
    for(unsigned int i = 0; i < length_steps; ++i)
        if (text[i] == ',') l_number_values++;
    int *steps = calloc(l_number_values, sizeof(int));
    for(unsigned i = 0; i < l_number_values; ++i){
        int step    = atoi(text);
        text = strchr(text, ',') + 1;
        steps[i] = step;
    }
    if (number_values)
        *number_values = l_number_values;
    return steps;
}

unsigned cfg_option_get_number_values(struct cfg_type_object* config, char *option)
{
    char *text = cfg_option_find(config, option);
    if(!text) return 0;
    unsigned int length_steps = strlen(text);
    int l_number_values = 1;
    for(unsigned int i = 0; i < length_steps; ++i)
        if (text[i] == ',') l_number_values++;
    return l_number_values;
}

int cfg_parse_object(file_line *line, cfg_create_config** r_options){
    if (line->line[0]=='['){
        for (size_t i = 1; i < line->size; i++){
            if (line->line[i]==']'){
                size_t j = 1;
                size_t add_size = sizeof(struct cfg_create_type_object) + sizeof(struct cfg_type_object_value) + i;
                *r_options = realloc(*r_options, r_options[0]->size_struct + add_size + sizeof(cfg_reader));
                cfg_create_config* options = *r_options;
                options->number_options++;

                struct cfg_create_type_object* ptr_cfg_type_object = (struct cfg_create_type_object *) (((char *) &options[1]) +
                        options->size_struct);
                ptr_cfg_type_object->number_types = 1;
                ptr_cfg_type_object->offset_next_type_object = add_size;
                struct cfg_type_object_value* ptr_cfg_type_object_value = (struct cfg_type_object_value*)&ptr_cfg_type_object[1];
                ptr_cfg_type_object_value->size_value = i;
                ptr_cfg_type_object_value->type = CFG_OBJECT;
                for (; j < i; j++)
                    ptr_cfg_type_object_value->value[j - 1] = line->line[j];
                ptr_cfg_type_object_value->value[j - 1] = 0;
                options->size_struct += add_size;
                return 1;
            }

        }
    }
    return 0;
}
int cfg_parse_type(file_line *line, cfg_create_config** r_options){
    if (line->line[0]!='#'){
        size_t size_name = 0;
        size_t start_offset_value = 0;
        size_t size_value = 0;
        for (size_t i = 1; i < line->size; i++){
            if (line->line[i]=='='){
                size_name = i;
                start_offset_value = i + 1;
            }
            if (start_offset_value){
                size_value = line->size - start_offset_value - 1;
                i = line->size;
            }

        }
        if (size_name == 0 || size_value == 0) return 0;
        size_t i = 0, j = 0;
        size_t add_size = 2 * sizeof(struct cfg_type_object_value) + size_name + 1 + size_value + 1;

        *r_options = realloc(*r_options, r_options[0]->size_struct + add_size + sizeof(cfg_reader));
        cfg_create_config* options = *r_options;
        struct cfg_type_object_value* last_ptr_cfg_type_object_value_0 = (struct cfg_type_object_value*)(((char*)options) + sizeof(cfg_create_config) + options->size_struct);
        struct cfg_type_object_value* last_ptr_cfg_type_object_value_1 = (struct cfg_type_object_value*)(((char*)options) + sizeof(cfg_create_config) + options->size_struct + size_name + 1 + sizeof(struct cfg_type_object_value));
        {
            struct cfg_create_type_object* object = (struct cfg_create_type_object *) &(options[1]);
            for (size_t i_option = 0; i_option < options->number_options - 1; object = (struct cfg_create_type_object *) (
                    (char *) object + object->offset_next_type_object), i_option++) {
            }
            object->number_types += 2 * 1;
            object->offset_next_type_object += add_size;
        }

        last_ptr_cfg_type_object_value_0->size_value = size_name + 1;
        last_ptr_cfg_type_object_value_0->type = CFG_TYPE_NAME;


        for (; i < size_name; i++)
            last_ptr_cfg_type_object_value_0->value[i] = line->line[i];
        last_ptr_cfg_type_object_value_0->value[i] = 0;

        last_ptr_cfg_type_object_value_1->size_value = size_value + 1;
        last_ptr_cfg_type_object_value_1->type = CFG_TYPE_VALUE;

        for (i = start_offset_value, j = 0; j < size_value; i++, j++)
            last_ptr_cfg_type_object_value_1->value[j] = line->line[i];
        last_ptr_cfg_type_object_value_1->value[j] = 0;

        options->size_struct += add_size;
        return 1;
    }
    return 0;
}
cfg_reader *read_cfg_malloc(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    cfg_create_config* options = calloc(1, sizeof(cfg_reader));
    file_line *line;
    while((line=file_getline(file)) != 0){
        file_strip(line);
        if (!cfg_parse_object(line, &options)){
            cfg_parse_type(line, &options);
        }
        free(line);
    }
    cfg_reader* return_options = (cfg_reader *) options;
    size_t i = 0;
    struct cfg_type_object* ptr_return_options = (struct cfg_type_object *)&(options[1]);
    for (struct cfg_create_type_object* object = (struct cfg_create_type_object *) &(options[1]); i < options->number_options; i++){
        size_t j = 0;
        ptr_return_options = (struct cfg_type_object *) object;
        ptr_return_options->current_type_object_value = (struct cfg_type_object_value *) &object[1];
        for (struct cfg_type_object_value* value = (struct cfg_type_object_value *) &object[1]; j < object->number_types; value = (struct cfg_type_object_value *) (
                (char *) value + value->size_value + sizeof(struct cfg_type_object_value)), j++){
            printf("%s\n", value->value);
            if (j + 1 < object->number_types){
                value->next = (struct cfg_type_object_value *) (
                        (char *) value + value->size_value + sizeof(struct cfg_type_object_value));
            } else
                value->next = NULL;
        }
        if (i + 1 < options->number_options){
            ptr_return_options->next_type_object = (struct cfg_type_object *)(
                    (char *) object + object->offset_next_type_object);
        }
                else
            ptr_return_options->next_type_object = NULL;
        object = (struct cfg_create_type_object *) ptr_return_options->next_type_object;
    }

    fclose(file);
    return return_options;
}
void read_cfg_free(cfg_reader *config){
    if (config) free(config);
    config = NULL;
}
