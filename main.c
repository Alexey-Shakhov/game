#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <GLFW/glfw3.h>

#include "render.h"
#include "utils.h"
#include "alloc.h"

static GLFWwindow* create_window()
{
    // TODO handle errors
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
     
    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
     
    return glfwCreateWindow(
            mode->width, mode->height, "Demo", monitor, NULL);
}

static GLFWwindow* destroy_window(GLFWwindow* window)
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

void render_test(GLFWwindow* window)
{
    Render* render = render_init(window);

    Vertex vertices[3] = {
        {
            .position = {2.0, 1.0, 20.0},
            .color = {1.0, 1.0, 1.0},
        },
        {
            .position = {1.0, 1.0, 20.0},
            .color = {1.0, 1.0, 1.0},
        },
        {
            .position = {3.0, 3.0, 20.0},
            .color = {1.0, 1.0, 1.0},
        },
    };
    uint16_t indices[3] = {
        0, 1, 2
    };
    render_upload_map_mesh(render, vertices, 3, indices, 3);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        render_draw_frame(render, window);
    }

    render_destroy(render);
}

int main()
{
    mem_init(MBS(24));


    GLFWwindow* window = create_window(&window);

    render_test(window);
    mem_check();

    destroy_window(window);
    
    mem_shutdown();

    printf("%s", "Success!\n");
    return EXIT_SUCCESS;
}
