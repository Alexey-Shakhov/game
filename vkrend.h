#ifndef VKREND_H
#define VKREND_H

#include <cglm/cglm.h>
typedef struct VkRender VkRender;
typedef struct Vertex {
    vec3 position;
    vec3 color;
} Vertex;
VkRender* render_init();
void render_loop(VkRender* self);
void render_destroy(VkRender* self);
void render_upload_map_mesh(
        VkRender* self, Vertex* vertices, size_t vertex_count,
        uint16_t* indices, size_t index_count);

#endif
