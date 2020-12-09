#define CGLM_DEFINE_PRINTS
#include "globals.h"
#include "alloc.h"
#include "utils.h"
#include "render.h"

#include "cglm/cglm.h"

#define MOVEMENT_SPEED 10
#define ROTATION_SPEED 0.001

int main()
{
    mem_init(MBS(24));

    Render* render = render_init();
    render_upload_map_mesh(render);

    vec3 cam_pos = {0.0f, 0.0f, 0.0f};
    vec3 cam_dir = {1.0f, 0.0f, 0.0f};
    vec3 cam_up = {0.0f, 0.0f, 1.0f};
    double now = glfwGetTime();
    double mouse_x;
    double mouse_y;
    glfwGetCursorPos(g_window, &mouse_x, &mouse_y);
    // Main loop
    while (!render_exit(render)) {
        if (glfwGetKey(g_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

        double elapsed = glfwGetTime() - now;
        if (elapsed < 0.01) continue;
        now = glfwGetTime();

        vec3 side;
        glm_vec3_cross(cam_dir, cam_up, side);

        vec2 delta = {0.0f, 0.0f};
        if (glfwGetKey(g_window, GLFW_KEY_W) == GLFW_PRESS) {
            delta[0] = 1.0f;
        } else if (glfwGetKey(g_window, GLFW_KEY_S) == GLFW_PRESS) {
            delta[0] = -1.0f;
        }
        if (glfwGetKey(g_window, GLFW_KEY_A) == GLFW_PRESS) {
            delta[1] = -1.0f;
        } else if (glfwGetKey(g_window, GLFW_KEY_D) == GLFW_PRESS) {
            delta[1] = 1.0f;
        }
        glm_vec2_normalize(delta);
        glm_vec2_scale(delta, MOVEMENT_SPEED * elapsed, delta);

        vec3 delta_dir;
        glm_vec3_scale(cam_dir, delta[0], delta_dir);
        vec3 delta_side;
        glm_vec3_scale(side, delta[1], delta_side);
        glm_vec3_add(cam_pos, delta_dir, cam_pos);
        glm_vec3_add(cam_pos, delta_side, cam_pos);

        double mouse_x_new;
        double mouse_y_new;
        glfwGetCursorPos(g_window, &mouse_x_new, &mouse_y_new);
        double mouse_dx = mouse_x_new - mouse_x;
        double mouse_dy = mouse_y_new - mouse_y;
        glfwGetCursorPos(g_window, &mouse_x, &mouse_y);
        // TODO also rotate up
        glm_vec3_rotate(cam_dir, -mouse_dx * ROTATION_SPEED, cam_up);
        glm_vec3_rotate(cam_dir, -mouse_dy * ROTATION_SPEED, side);

        render_draw_frame(render, cam_pos, cam_dir, cam_up);
    }

    render_destroy(render);

    mem_check();
    mem_inspect();
    mem_shutdown();
    printf("%s", "Success!\n");
    return EXIT_SUCCESS;
}
