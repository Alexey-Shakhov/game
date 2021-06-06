#include "collision.h"
#define POLE_SPACING 0.1

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

float get_height(Vertex* vertices, uint32_t vertex_count, uint16_t* indices, uint32_t index_count, float x, float y)
{
    float z_highest = -1000.0;
    bool ground_found = false;
    int tricount=0;
    for (size_t index = 0; index < index_count; index += 3) {
        tricount++;
        vec2 p = {x, y};
        vec2 a = {vertices[indices[index]].position[0], vertices[indices[index]].position[1]};
        vec2 b = {vertices[indices[index+1]].position[0], vertices[indices[index+1]].position[1]};
        vec2 c = {vertices[indices[index+2]].position[0], vertices[indices[index+2]].position[1]};
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

        if ((u >= 0.0) && (v >= 0.0) && (u + v <= 1.0)) {
//            printf("koo\n");
            float az = vertices[indices[index]].position[2];
            float z = az + (vertices[indices[index+1]].position[2] - az) * v +
                                        (vertices[indices[index+2]].position[2] - az) * u;
            if (z > z_highest) z_highest = z;
            ground_found = true;
        } else {
 //           printf("nope\n");
        }
    }
    float z;
    if (ground_found) z = z_highest; else z = 0.0;
    printf("tri count %d\n", tricount);
    return z;
}
