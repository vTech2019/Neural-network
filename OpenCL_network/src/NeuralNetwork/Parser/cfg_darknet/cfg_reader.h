//
// Created by human on 05.03.2020.
//

#ifndef OPENCL_CFG_READER_H
#define OPENCL_CFG_READER_H

#include "../../File/file.h"

enum cfg_type_enum{
    CFG_OBJECT = 1,
    CFG_TYPE_NAME,
    CFG_TYPE_VALUE,
    CFG_TYPE_USED,
};
struct cfg_type_object_value{
    struct cfg_type_object_value* next;
    enum cfg_type_enum type;
    unsigned int size_value;
    char value[0];
};
struct cfg_create_type_object{
    size_t number_types;
    size_t reserved;
    size_t offset_next_type_object;
};
typedef struct{
    size_t size_struct;
    size_t number_options;
}cfg_create_config;

struct cfg_type_object{
    size_t number_types;
    struct cfg_type_object_value* current_type_object_value;
    struct cfg_type_object* next_type_object;
};
typedef struct cfg_config{
    size_t size_struct;
    size_t number_options;
    struct cfg_type_object objects[0];
}cfg_reader;

cfg_reader *read_cfg_malloc(char *filename);
void read_cfg_free(cfg_reader *config);

int cfg_option_find_int(struct cfg_type_object* config, char *option, int default_value);
float cfg_option_find_float(struct cfg_type_object* config, char *option, float default_value);
char* cfg_option_find_string(struct cfg_type_object* config, char *option, char* default_value);
int* cfg_option_find_ints_malloc(struct cfg_type_object* config, char *option, int* number_values);
float* cfg_option_find_floats_malloc(struct cfg_type_object* config, char *option, int* number_values);

unsigned cfg_option_get_number_values(struct cfg_type_object* config, char *option);

unsigned cfg_option_find_floats_return_number_write(struct cfg_type_object* config, char *option, float return_values[], unsigned number_values);
unsigned cfg_option_find_ints_return_number_write(struct cfg_type_object* config, char *option, int return_values[], unsigned number_values);
#endif //OPENCL_CFG_READER_H
