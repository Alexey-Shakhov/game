#ifndef RENDER_H
#define RENDER_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cglm/cglm.h>

typedef struct Render Render;
Render* render_init();
bool render_exit(Render* render);
void render_draw_frame(Render* self, vec3 cam_pos, vec3 cam_dir, vec3 cam_up);
void render_destroy(Render* self);

typedef struct Vertex {
    vec3 position;
    vec2 tex_coord;
    vec3 normal;
} Vertex;
void load_scene(Render* self);

#endif
