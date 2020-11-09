#ifndef ALLOC_H
#define ALLOC_H

#include <stddef.h>

void mem_init(size_t size);
void* mem_alloc(size_t bytes);
void mem_free(void* p);
void mem_shutdown();

#ifndef RELEASE
void mem_check();
#endif

#endif
