#ifndef RENDER_H
#define RENDER_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cglm/cglm.h>

typedef struct Render Render;
Render* render_init();
void render_draw_frame(Render* self, GLFWwindow* window);
void render_destroy(Render* self);

typedef struct Vertex {
    vec3 position;
    vec3 color;
    vec2 tex_coord;
} Vertex;
void render_upload_map_mesh(Render* self);

#endif
