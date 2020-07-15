#ifndef PLATFORM_H
#define PLATFORM_H

#include <cglm/cglm.h>

typedef struct Render Render;
typedef struct Vertex {
    vec3 position;
    vec3 color;
} Vertex;
Render* render_init();
void render_loop(Render* self);
void render_destroy(Render* self);
void render_upload_map_mesh(
        Render* self, Vertex* vertices, size_t vertex_count,
        uint16_t* indices, size_t index_count);

#endif
