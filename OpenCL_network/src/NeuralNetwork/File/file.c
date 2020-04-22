//
// Created by human on 05.03.2020.
//

#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include "file.h"

void file_error(char *s){
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}
void malloc_error(char* file_name, size_t file_line){
    fprintf(stderr, "Malloc error FILE: %s; LINE %ld\n", file_name, file_line);
    exit(0);
}

void file_strip(file_line* string){
    size_t offset = 0;
    for(size_t i = 0; i < string->size; ++i){
        char c = string->line[i];
        if(c==' '||c=='\t'||c=='\n') ++offset;
        else string->line[i-offset] = c;
    }
    string->line[string->size-offset] = '\0';
}

file_line* file_getline(FILE *fp){
    if(feof(fp)) return 0;
    file_line* line = NULL;
    size_t size = 1;
    size_t current = 0;
    do{
        if(current == size - 1){
            size *= 2;
            if ((line = realloc(line, size + sizeof(file_line))) == NULL)
                malloc_error(__FILE__, __LINE__);
        }
        size_t readsize = size - current;
        if(readsize > INT_MAX) readsize = INT_MAX - 1;
        if(!fgets(line->line + current, readsize, fp)){
            free(line);
            return 0;
        }
        current = strlen(line->line);
    } while((line->line[current-1] != '\n') && !feof(fp));
    if(line->line[current-1] == '\n') line->line[current-1] = '\0';
    line->size = current;

    return line;
}