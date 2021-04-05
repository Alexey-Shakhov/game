#include <stdlib.h>
#include <inttypes.h>
#include "alloc.h"
#include "utils.h"

#define MINFRAGMENT 64
#define ZONEID 0xdeadbeef
#define ALIGNMENT 16

typedef struct Memblock Memblock;
typedef struct Memblock {
    uint64_t size; // Including the struct
    Memblock* prev;
    Memblock* next;
    uint32_t id;
    uint32_t tag; // 0 - free, else: used
} Memblock;

typedef struct Memzone {
    uint64_t size; // Including the struct
    Memblock link;
    Memblock* rover;
} Memzone;

void zone_init(Memzone* zone, size_t size)
{
    Memblock* block;
    zone->link.next = zone->link.prev =
            block = (Memblock*)((char*) zone + sizeof(Memzone));
    zone->link.id = 0;
    zone->link.size = 0;
    zone->link.tag = 1;
    zone->rover = block;
    zone->size = size;

    block->prev = block->next = &zone->link;
    block->id = ZONEID;
    block->tag = 0;
    block->size = size - sizeof(Memzone);
}

static Memzone* mainzone;

void mem_init(size_t size)
{
    mainzone = (Memzone*) malloc(size);
    if (!mainzone) fatal("Failed to allocate game memory.");
    zone_init(mainzone, size);
}

void* mem_alloc(size_t size)
{
    size += sizeof(Memblock);
    size = (size + ALIGNMENT - 1) & ~ (ALIGNMENT - 1); 
    Memblock* block = mainzone->rover;
    Memblock* start = block->prev;
    while (block->tag || block->size < size) {
        if (block == start) return NULL;
        block = block->next;
    }
    mainzone->rover = block->next;

    uint64_t extra = block->size - size;
    if (extra >= MINFRAGMENT) {
        Memblock* new_block = (Memblock*) ((uint8_t*) block + size);
        new_block->size = extra;
        new_block->tag = 0;
        new_block->prev = block;
        new_block->id = ZONEID;
        new_block->next = block->next;
        new_block->next->prev = new_block;

        block->next = new_block;
        block->size = size;
    }
    
    block->tag = 1;
    block->id = ZONEID;
    return (void*) ((char*) block + sizeof(Memblock));
}

void mem_free(void* ptr)
{
    if (!ptr) fatal("Trying to free a NULL pointer.");

    Memblock* block = (Memblock*) ((char*) ptr - sizeof(Memblock));
    if (block->id != ZONEID) fatal("Trying to free a pointer without ZONEID.");
    if (block->tag == 0) fatal("Trying to free a free pointer.");

    block->tag = 0;

    Memblock* other = block->prev;
    if (!other->tag) {
        other->size += block->size;
        other->next = block->next;
        other->next->prev = other;
        if (block == mainzone->rover) mainzone->rover = other;
        block = other;
    }

    other = block->next;
    if (!other->tag) {
        block->size += other->size;
        block->next = other->next;
        block->next->prev = block;
        if (other == mainzone->rover) mainzone->rover = block;
    }
}

void mem_shutdown()
{
    free(mainzone);
}

#ifndef RELEASE
void mem_check()
{
    for (Memblock* block = mainzone->link.next; ; block = block->next) {
        if (block->next == &mainzone->link) break;

        if ((char*) block + block->size != (char*) block->next)
            fatal("MEMCHECK: block size does not touch the next block.\n");

        if (block->next->prev != block)
            fatal("MEMCHECK: next block doesn't have proper back link.\n");

        if (!block->tag && !block->next->tag)
            fatal("MEMCHECK: two consecutive free blocks.\n");
    }
}

void mem_inspect()
{
    Memblock* block = mainzone->link.next;
    printf("-----------------MEMORY REPORT START-----------------\n");
    while (block != &mainzone->link) {
        printf("%f\n", ((float)block->size) / 1024 / 1024);
        block = block->next;
    }
    printf("------------------MEMORY REPORT END------------------\n\n");
}
#endif
