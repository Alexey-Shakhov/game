#define CGLM_DEFINE_PRINTS
#include "globals.h"
#include "alloc.h"
#include "utils.h"
#include "scene.h"
#include "render.h"

#include "cglm/cglm.h"

#define MOVEMENT_SPEED 10
#define ROTATION_SPEED 0.001

#define MLOOK_LIMIT (CGLM_PI/16)
#define OBJECT_MOVE_SPEED 0.01

struct EdState {
    bool lmb_pressed;
    bool rmb_pressed;
    Node* sel_object;
    Light* sel_light;
};

struct EdState ed_state = {
    .lmb_pressed = false,
    .sel_object = NULL,
    .sel_light = NULL,
};

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        ed_state.lmb_pressed = true;
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        ed_state.rmb_pressed = true;
    }
}

int main()
{
    mem_init(MBS(24));

    render_init();
    load_scene();

    vec3 cam_pos = {0.0f, 0.0f, 0.0f};
    vec3 cam_dir = {1.0f, 0.0f, 0.0f};
    vec3 cam_up = {0.0f, 0.0f, 1.0f};
    double now = glfwGetTime();
    double mouse_x;
    double mouse_y;
    glfwGetCursorPos(g_window, &mouse_x, &mouse_y);
    glfwSetMouseButtonCallback(g_window, mouse_button_callback);
    // Main loop
    while (!render_exit()) {
        glfwPollEvents();
        if (glfwGetKey(g_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) break;

        double elapsed = glfwGetTime() - now;
        if (elapsed < 0.01) continue;
        now = glfwGetTime();

        // Get cursor
        double mouse_x_new;
        double mouse_y_new;
        glfwGetCursorPos(g_window, &mouse_x_new, &mouse_y_new);
        double mouse_dx = mouse_x_new - mouse_x;
        double mouse_dy = mouse_y_new - mouse_y;
        mouse_x = mouse_x_new;
        mouse_y = mouse_y_new;

        vec3 side;
        glm_vec3_cross(cam_dir, cam_up, side);

        if (ed_state.lmb_pressed) {
            ed_state.lmb_pressed = false;
            uint32_t code = get_object_code(1366/2, 768/2);
            if (code > 0) {
                ed_state.sel_object = &scene.nodes[code-1];
            }
        }

        if (ed_state.rmb_pressed) {
            ed_state.rmb_pressed = false;
            ed_state.sel_object = NULL;
        }

        if (ed_state.sel_object) {
            vec3 current_up;
            glm_vec3_cross(cam_dir, side, current_up);
            vec3 offset_up;
            glm_vec3_scale(current_up, OBJECT_MOVE_SPEED * mouse_dy, offset_up);
            vec3 offset_side;
            glm_vec3_scale(side, OBJECT_MOVE_SPEED * mouse_dx, offset_side);
            glm_translate(ed_state.sel_object->transform, offset_up);
            glm_translate(ed_state.sel_object->transform, offset_side);
        } else {
            // Mouselook
            glm_vec3_rotate(cam_dir, -mouse_dx * ROTATION_SPEED, cam_up);
            float mlook_angle = -mouse_dy * ROTATION_SPEED;
            float up_cam_angle = glm_vec3_angle(cam_up, cam_dir);
            if (mlook_angle > up_cam_angle - MLOOK_LIMIT)
                mlook_angle = up_cam_angle - MLOOK_LIMIT;
            if (-mlook_angle > CGLM_PI - up_cam_angle - MLOOK_LIMIT)
                mlook_angle = - (CGLM_PI - up_cam_angle - MLOOK_LIMIT);
            glm_vec3_rotate(cam_dir, mlook_angle, side);
        }

        // Keyboard movement
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

        render_draw_frame(cam_pos, cam_dir, cam_up);
    }

    render_destroy();

    mem_check();
    mem_inspect();
    mem_shutdown();
    printf("%s", "Success!\n");
    return EXIT_SUCCESS;
}
