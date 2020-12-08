#include "globals.h"
#include "alloc.h"
#include "utils.h"
#include "render.h"

#include "cglm/cglm.h"

#define MOVEMENT_SPEED 2.5

int main()
{
    mem_init(MBS(24));

    Render* render = render_init();
    render_upload_map_mesh(render);

    vec3 cam_pos = {0.0f, 0.0f, 0.0f};
    vec3 cam_dir = {1.0f, 0.0f, 0.0f};
    double now = glfwGetTime();
    // Main loop
    while (!render_exit(render)) {
        double elapsed = glfwGetTime() - now;
        if (elapsed < 0.01) continue;

        now = glfwGetTime();
        if (glfwGetKey(g_window, GLFW_KEY_W) == GLFW_PRESS) {
            cam_pos[0] += MOVEMENT_SPEED * elapsed;
        }
        render_draw_frame(render);
    }

    render_destroy(render);

    mem_check();
    mem_inspect();
    mem_shutdown();
    printf("%s", "Success!\n");
    return EXIT_SUCCESS;
}
