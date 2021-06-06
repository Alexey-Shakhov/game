#ifndef COLLISION_H
#define COLLISION_H

#define CGLM_DEFINE_PRINTS
#include <cglm/cglm.h>
#include <stdbool.h>
#include "scene.h"

bool point_in_triangle(vec2 a, vec2 b, vec2 c, vec2 p);
float get_height(Vertex* vertices, uint32_t vertex_count, uint16_t* indices, uint32_t index_count, float x, float y);

#endif
