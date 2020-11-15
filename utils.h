#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <stdio.h>

void errprint(const char* const err);
void fatal(const char* const err);
void* malloc_nofail(size_t bytes);
int read_binary_file(const char *filename, char* *const o_dest, size_t *o_size);

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define KBS(num) (num*1024)
#define MBS(num) (num*1024*1024)
#define GBS(num) (num*1024*1024*1024)

#ifndef RELEASE
#define DBASSERT(condition) { \
    if (!(condition)) { \
        printf("Assertion failed. File: %s. Line: %d\n", __FILE__, __LINE__);\
        exit(EXIT_FAILURE); \
    }                                           \
} 
#else
#define DBASSERT(condition)
#endif

#endif
