#include "utils.h"
#include "alloc.h"

#include <stdio.h>

void errprint(const char* const err) {
    fprintf(stderr, "%s", err);
}

void fatal(const char* const err) {
    fprintf(stderr, "%s", err);
    exit(EXIT_FAILURE);
}

void* malloc_check(size_t bytes) {
    void* ptr = mem_alloc(bytes);
    if (!ptr) {
        errprint("Failed to allocate memory.");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

int read_binary_file(const char *filename, char* *const o_dest, size_t *o_size)
{   
    FILE *file = fopen(filename, "rb");
    if (!file)
        return 1;
    
    fseek(file, 0, SEEK_END);
    *o_size = ftell(file);
    rewind(file);
    
    *o_dest = malloc_check(*o_size);
    fread(*o_dest, *o_size, 1, file);

    fclose(file);

    return 0;
}   
