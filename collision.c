#include "collision.h"
#define POLE_SPACING 0.5

bool point_in_triangle(vec2 a, vec2 b, vec2 c, vec2 p)
{
    vec2 ab;
    vec2 ac;
    vec2 ap;
    glm_vec2_sub(b, a, ab);
    glm_vec2_sub(c, a, ac);
    glm_vec2_sub(p, a, ap);

    float cc = glm_vec2_dot(ac, ac);
    float bc = glm_vec2_dot(ab, ac);
    float pc = glm_vec2_dot(ap, ac);
    float bb = glm_vec2_dot(ab, ab);
    float pb = glm_vec2_dot(ab, ap);

    float denom = cc * bb - bc * bc;
    float u = (bb * pc - bc * pb) / denom;
    float v = (cc * pb - bc * pc) / denom;
    return (u >= 0) && (v >= 0) && (u + v < 1);
}

float get_height(Vertex* vertices, uint32_t vertex_count, float x, float y)
{
    for (Vertex* vert = vertices; vert < vertices + vertex_count; vert += 3) {
        vec2 p = {x, y};
        vec2 a = {vert[0].position[0], vert[0].position[1]};
        vec2 b = {vert[1].position[0], vert[1].position[1]};
        vec2 c = {vert[2].position[0], vert[2].position[1]};
        vec2 ab;
        vec2 ac;
        vec2 ap;
        glm_vec2_sub(b, a, ab);
        glm_vec2_sub(c, a, ac);
        glm_vec2_sub(p, a, ap);

        float cc = glm_vec2_dot(ac, ac);
        float bc = glm_vec2_dot(ab, ac);
        float pc = glm_vec2_dot(ac, ap);
        float bb = glm_vec2_dot(ab, ab);
        float pb = glm_vec2_dot(ab, ap);

        float denom = cc * bb - bc * bc;
        float u = (bb * pc - bc * pb) / denom;
        float v = (cc * pb - bc * pc) / denom;

        if ((u >= 0.0) && (v >= 0.0) && (u + v < 1.0)) {
            float az = vert[0].position[2];
            float z = az + (vert[1].position[2] - az) * v +
                                        (vert[2].position[2] - az) * u;
            printf("z: %f\n", z);
            return z;
        }
    }
}
