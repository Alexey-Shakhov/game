#ifndef SCENE_H
#define SCENE_H

#include "render.h"
#include "vkhelpers.h"

typedef struct Primitive {
    uint32_t texture_id;
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t index_count;
} Primitive;

typedef struct Mesh {
    Primitive* primitives;
    uint32_t primitives_count;
} Mesh;
void destroy_mesh(Mesh* mesh);

typedef struct Node Node;
typedef struct Node {
    Node* parent;
    Node** children;
    uint32_t children_count;
    vec3 translation;
    versor rotation;
    vec3 scale;
    Mesh* mesh;
    uint32_t id;
} Node;

void node_make_matrix(Node* node, mat4 dest);

typedef struct Light {
    vec3 pos;
    uint32_t code;
    vec3 color;
    uint32_t pad;
} Light;
#define LIGHT_COUNT 2

typedef struct Vertex {
    vec3 position;
    vec2 tex_coord;
    vec3 normal;
} Vertex;

typedef struct Scene {
    Mesh* meshes;
    size_t mesh_count;
    Node* nodes;
    size_t node_count;
    Texture* textures;
    size_t texture_count;
    Light* lights;
    size_t light_count;

    Vertex* vertices;
    size_t vertex_count;
    uint16_t* indices;
    size_t index_count;

    Buffer vertex_buffer;
    Buffer index_buffer;
    Buffer lights_buffer;
} Scene;

void destroy_scene(Scene* scene);

extern Scene scene;

#endif
