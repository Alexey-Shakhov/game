#include "vkrend.h"
#include "alloc.h"
#include <assert.h>
#include "utils.h"

void render_test()
{
    VkRender* render = render_init();

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

    render_loop(render);
    render_destroy(render);
}

int main()
{
    mem_init(1024);

    void* pt3 = mem_alloc(68);
    void* pt4 = mem_alloc(63);
    void* pt2 = mem_alloc(69);
    void* ptr = mem_alloc(167);
    void* pt5 = mem_alloc(47);

    mem_inspect();
    mem_free(ptr);
    mem_free(pt3);
    mem_free(pt5);
    mem_free(pt2);
    mem_free(pt4);
    mem_inspect();

    printf("%s", "Success!\n");
    return EXIT_SUCCESS;
}
