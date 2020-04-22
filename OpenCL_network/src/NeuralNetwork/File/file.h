//
// Created by human on 05.03.2020.
//

#ifndef OPENCL_FILE_H
#define OPENCL_FILE_H
#include <stdio.h>
typedef struct{
    size_t size;
    char line[0];
}file_line;

void file_error(char *s);
file_line* file_getline(FILE *fp);
void file_strip(file_line* string);


#endif //OPENCL_FILE_H
