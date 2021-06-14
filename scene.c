#include "alloc.h"
#include "globals.h"
#include "scene.h"

void destroy_mesh(Mesh* mesh)
{
    mem_free(mesh->primitives);
}

void node_make_matrix(Node* node, mat4 dest)
{
    glm_mat4_identity(dest);
    glm_translate(dest, node->translation);
    glm_quat_rotate(dest, node->rotation, dest);
    glm_scale(dest, node->scale);
}

void destroy_scene(Scene* scene)
{
    for (size_t i=0; i < scene->node_count; i++) mem_free(scene->nodes[i].children);
    mem_free(scene->nodes);
    for (size_t i=0; i < scene->mesh_count; i++) {
        destroy_mesh(&scene->meshes[i]);
    }
    mem_free(scene->meshes);
    mem_free(scene->lights);

    mem_free(scene->vertices);
    mem_free(scene->indices);
}

Scene scene;
