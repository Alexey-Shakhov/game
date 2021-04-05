#ifndef RENDER_H
#define RENDER_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cglm/cglm.h>

void render_init();
bool render_exit();
void render_draw_frame(vec3 cam_pos, vec3 cam_dir, vec3 cam_up);
void render_destroy();
uint32_t get_object_code(uint32_t x, uint32_t y);

typedef struct Vertex {
    vec3 position;
    vec2 tex_coord;
    vec3 normal;
} Vertex;
void load_scene();

typedef struct Buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
} Buffer;

void destroy_buffer(Buffer* buffer);

#define LIGHT_ID_OFFSET 100000

#endif
