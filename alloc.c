// TODO maybe add counters of allocations and frees to report double frees?

#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include "alloc.h"
#include "utils.h"

#define ALIGNMENT 16
#define CHUNK_HEADER_SIZE 32 /* Minimum size that can contain ChunkHeader
                                and divides by alignment */

typedef struct ChunkHeader ChunkHeader;
typedef struct ChunkHeader {
    size_t size;
    ChunkHeader* prev;
    bool free;
} ChunkHeader;

struct Allocator {
    uint8_t* ptr;
    size_t size;
} allocator;

void mem_init(size_t size)
{
    allocator.ptr = malloc(size);
    allocator.size = size;
    DBASSERT((size_t) allocator.ptr % ALIGNMENT == 0);
    if (!allocator.ptr) {
        errprint("Failed to allocate game memory.");
        exit(EXIT_FAILURE);
    }

    ChunkHeader* first_chunk = (ChunkHeader*) allocator.ptr;
    first_chunk->size = size - CHUNK_HEADER_SIZE;
    first_chunk->free = true;
    first_chunk->prev = NULL;
}

void* mem_alloc(size_t bytes)
{
    if (!bytes) return NULL;

    bytes = bytes + (ALIGNMENT - bytes & (ALIGNMENT - 1));

    ChunkHeader* chunk = (ChunkHeader*) allocator.ptr;
    while (true) {
        if ((uint8_t*) chunk >= allocator.ptr + allocator.size) return NULL;
        if (chunk->free && chunk->size >= bytes + CHUNK_HEADER_SIZE) break;
        chunk = (ChunkHeader*) ((uint8_t*) chunk + CHUNK_HEADER_SIZE + chunk->size);
    }
    uint8_t* alloc = (uint8_t*) chunk + CHUNK_HEADER_SIZE;

    ChunkHeader* new_chunk = (ChunkHeader*) (alloc + bytes);
    new_chunk->size = chunk->size - bytes - CHUNK_HEADER_SIZE;
    new_chunk->free = true;
    new_chunk->prev = chunk;
    chunk->free = false;
    chunk->size = bytes;
    
    return alloc;
}

void mem_free(void* p)
{
    ChunkHeader* chunk = (ChunkHeader*) ((uint8_t*) p - CHUNK_HEADER_SIZE);
    chunk->free = true;

    ChunkHeader* next = (ChunkHeader*)
        ((uint8_t*) chunk + CHUNK_HEADER_SIZE + chunk->size);
    bool merged_with_next = false;
    if ((uint8_t*) next < allocator.ptr + allocator.size && next->free) {
        merged_with_next = true;
        chunk->size = chunk->size + CHUNK_HEADER_SIZE + next->size;
    }

    if (chunk->prev && chunk->prev->free) {
        chunk->prev->size = chunk->prev->size + CHUNK_HEADER_SIZE + chunk->size;
        if (merged_with_next) {
            ChunkHeader* next = (ChunkHeader*)
                ((uint8_t*) chunk + CHUNK_HEADER_SIZE + chunk->size);
        }
        if ((uint8_t*) next < allocator.ptr + allocator.size) {
            next->prev = chunk->prev;
        }
    }
}

void mem_shutdown()
{
#ifndef RELEASE
    ChunkHeader* first_chunk = (ChunkHeader*) allocator.ptr;
    DBASSERT(first_chunk->free);
    DBASSERT(first_chunk->prev == NULL);
    DBASSERT(first_chunk->size == allocator.size - CHUNK_HEADER_SIZE);
#endif
    free(allocator.ptr);
}

#ifndef RELEASE
void mem_inspect()
{
    ChunkHeader* chunk = (ChunkHeader*) allocator.ptr;
    printf("---Memory report---\n");
    while ((uint8_t*) chunk < allocator.ptr + allocator.size) {
        if (chunk->free) printf("Free. "); else printf("Taken. ");
        printf("Relative address: %d. Size: %d.\n", (uint8_t*) chunk - allocator.ptr, chunk->size);
        chunk = (ChunkHeader*) ((uint8_t*) chunk + CHUNK_HEADER_SIZE + chunk->size);
    }
    printf("\n");
}
#endif
