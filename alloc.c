#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include "alloc.h"
#include "utils.h"

#define ALIGNMENT 16
#define BLOCK_HEADER_SIZE 32 /* Minimum size that can contain Memblock
                                and divides by alignment */
#define MIN_BLOCK_SIZE 64 // DOOM value

#define USEDID 0xdeadbeef

typedef struct Memblock Memblock;
typedef struct Memblock {
    size_t size; // Including the header
    Memblock* prev;
    Memblock* next;
    uint32_t usedid; // Must be equal to USEDID if used
    bool used;
} Memblock;

struct Allocator {
    uint8_t* ptr;
    size_t size;
    Memblock link;
} allocator;

void mem_clear()
{
    Memblock* block;
    allocator.link.next = allocator.link.prev =
            block = (Memblock*) allocator.ptr;

    block->prev = block->next = &allocator.link;
    block->used = false;
    block->size = allocator.size;
}

void mem_init(size_t size)
{
    allocator.ptr = malloc(size);
    allocator.size = size;
    DBASSERT((size_t) allocator.ptr % ALIGNMENT == 0);
    if (!allocator.ptr) {
        errprint("Failed to allocate game memory.");
        exit(EXIT_FAILURE);
    }

    allocator.link.used = true;

    mem_clear();
}

void* mem_alloc(size_t bytes)
{
    bytes = bytes + (ALIGNMENT - bytes & (ALIGNMENT - 1));
    bytes += BLOCK_HEADER_SIZE;

    // TODO make faster with the rover
    Memblock* block = allocator.link.next;
    while (true) {
        if (block == &allocator.link) return NULL;
        if (block->size >= bytes && !block->used) break;
        block = block->next;
    }

    int extra = block->size - bytes;

    if (extra >= MIN_BLOCK_SIZE) {
        Memblock* new_block = (Memblock*) ((uint8_t*) block + bytes);
        new_block->size = extra;
        new_block->used = false;
        new_block->prev = block;
        new_block->next = block->next;
        new_block->next->prev = new_block;

        block->next = new_block;
        block->size = bytes;
    }
    
    block->used = true;
    block->usedid = USEDID;
    return (void*) ((uint8_t*) block + BLOCK_HEADER_SIZE);
}

void mem_free(void* ptr)
{
    Memblock* block = (Memblock*) ((uint8_t*) ptr - BLOCK_HEADER_SIZE);
    if (block->usedid != USEDID) {
        //TODO proper shutdown
        errprint("Double free or memory corruption.");
        exit(EXIT_FAILURE);
    }

    block->used = false;
    block->usedid = 0;

    Memblock* other = block->prev;

    if (!other->used) {
        other->size += block->size;
        other->next = block->next;
        other->next->prev = other;

        block = other;
    }

    other = block->next;
    if (!other->used) {
        block->size += other->size;
        block->next = other->next;
        block->next->prev = block;
    }
}

void mem_shutdown()
{
    free(allocator.ptr);
}

#ifndef RELEASE
// Proudly stolen from DOOM source code
void mem_check()
{
    Memblock* block;

    for (block = allocator.link.next; ; block = block->next) {
        if (block->next == &allocator.link) break;

        if ((uint8_t*) block + block->size != (uint8_t*) block->next)
            fatal("MEMCHECK: block size does not touch the next block.\n");

        if (block->next->prev != block)
            fatal("MEMCHECK: next block doesn't have proper back link.\n");

        if (!block->used && !block->next->used)
            fatal("MEMCHECK: two consecutive free blocks.\n");
    }
}
#endif
