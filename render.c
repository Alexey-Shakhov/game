#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define CGLM_DEFINE_PRINTS
#include <cglm/cglm.h>
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include "utils.h"
#include "alloc.h"

#include "render.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define APP_NAME "Demo"
#define ENGINE_NAME "None"

#define FRAMES_IN_FLIGHT 2

extern GLFWwindow* g_window;

VkPhysicalDevice g_physical_device;
VkDevice g_device;

// Scene structs
typedef struct Primitive {
    VkDescriptorSet texture;
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t index_count;
} Primitive;

typedef struct Mesh {
    Primitive* primitives;
    uint32_t primitives_count;
} Mesh;
void destroy_mesh(Mesh* mesh)
{
    mem_free(mesh->primitives);
}

typedef struct Node Node;
typedef struct Node {
    Node* parent;
    Node** children;
    uint32_t children_count;
    mat4 transform;
    Mesh* mesh;
} Node;

#define MAX_TEXTURES 50
typedef struct Texture {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    VkDescriptorSet desc_set;
} Texture;

void destroy_texture(Texture* texture)
{
    vkDestroyImageView(g_device, texture->view, NULL);
    vkDestroyImage(g_device, texture->image, NULL);
    vkFreeMemory(g_device, texture->memory, NULL);
}

typedef struct Light {
    vec3 pos;
    uint32_t pad;
    vec3 color;
    uint32_t pad2;
} Light;
#define LIGHT_COUNT 2

typedef struct Buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
} Buffer;

void destroy_buffer(Buffer* buffer)
{
    vkDestroyBuffer(g_device, buffer->buffer, NULL);
    vkFreeMemory(g_device, buffer->memory, NULL);
}

typedef struct Scene {
    Mesh* meshes;
    size_t mesh_count;
    Node* nodes;
    size_t node_count;
    Texture* textures;
    size_t texture_count;
    Light* lights;
    size_t light_count;

    Buffer vertex_buffer;
    Buffer index_buffer;
    Buffer lights_buffer;
} Scene;

void destroy_scene(Scene* scene)
{
    for (size_t i=0; i < scene->node_count; i++) mem_free(scene->nodes[i].children);
    mem_free(scene->nodes);
    for (size_t i=0; i < scene->mesh_count; i++) {
        destroy_mesh(&scene->meshes[i]);
    }
    mem_free(scene->meshes);
    for (size_t i=0; i < scene->texture_count; i++) {     
        destroy_texture(&scene->textures[i]);
    } 
    mem_free(scene->textures);
    mem_free(scene->lights);

    destroy_buffer(&scene->lights_buffer);
    destroy_buffer(&scene->index_buffer);
    destroy_buffer(&scene->vertex_buffer);
}

static Scene scene;


typedef struct MrtUbo {
    mat4 view_proj;
} MrtUbo;

typedef struct DeferredUbo {
    vec3 view_pos;
    uint32_t light_count;
} DeferredUbo;

typedef struct PushConstants {
    mat4 model;
} PushConstants;

static void create_2d_image(uint32_t width, uint32_t height,
        VkSampleCountFlagBits samples, VkFormat format, VkImageTiling tiling,
        VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
        VkImage* image, VkDeviceMemory* memory);
static void create_2d_image_view(VkImage image, VkFormat format,
        VkImageAspectFlags aspect_flags, VkImageView* image_view);
static VkCommandBuffer begin_one_time_command_buffer(VkCommandPool command_pool);
static void submit_one_time_command_buffer(VkQueue queue,
        VkCommandBuffer command_buffer, VkCommandPool command_pool);
static VkFormat find_depth_format();
static VkShaderModule create_shader_module(const char* path);
static int find_memory_type(
        VkMemoryRequirements memory_requirements,
        VkMemoryPropertyFlags required_properties
);
static int create_buffer(
        size_t size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, Buffer* buffer);
static void device_local_buffer_from_data(
        void* data,
        size_t size,
        VkBufferUsageFlags usage,
        VkQueue queue,
        VkCommandPool command_pool,
        Buffer* buffer
);
static void upload_to_device_local_buffer(
        void* data,
        size_t size,
        Buffer* destination,
        VkQueue queue,
        VkCommandPool command_pool
);
static Buffer upload_data_to_staging_buffer(void* data, size_t size);

enum { VALIDATION_ENABLED = 1 };

const char *const VALIDATION_LAYERS[] = {
    "VK_LAYER_KHRONOS_validation"
};

#define DEVICE_EXTENSION_COUNT 1
const char *const DEVICE_EXTENSIONS[DEVICE_EXTENSION_COUNT] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

typedef struct Attachment {
    VkImage image;
    VkImageView view;
    VkDeviceMemory memory;
} Attachment;

void create_attachment(Attachment* att, uint32_t width, uint32_t height,
        VkFormat format, VkImageUsageFlags usage)
{
    VkImageAspectFlags aspect_mask = 0;
    if (usage == VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        aspect_mask |= VK_IMAGE_ASPECT_DEPTH_BIT;
    } else {
        aspect_mask |= VK_IMAGE_ASPECT_COLOR_BIT;
    };
    usage |= VK_IMAGE_USAGE_SAMPLED_BIT;

    create_2d_image(width, height, VK_SAMPLE_COUNT_1_BIT, format,
        VK_IMAGE_TILING_OPTIMAL, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &att->image, &att->memory);
    create_2d_image_view(att->image, format, aspect_mask, &att->view);
}

void destroy_attachment(Attachment* att)
{
    vkDestroyImageView(g_device, att->view, NULL);
    vkDestroyImage(g_device, att->image, NULL);
    vkFreeMemory(g_device, att->memory, NULL);
}

typedef struct Render {
    VkInstance instance;
    VkSurfaceKHR surface;
    uint32_t graphics_family;
    uint32_t present_family;
    VkQueue graphics_queue;
    VkQueue present_queue;
    VkSwapchainKHR swapchain;
    VkFormat swapchain_format;
    VkExtent2D swapchain_extent;
    VkImage swapchain_images[FRAMES_IN_FLIGHT];
    VkImageView swapchain_image_views[FRAMES_IN_FLIGHT];
    VkRenderPass render_pass;
    VkRenderPass offscreen_render_pass;
    VkRenderPass lights_ui_render_pass;
    VkPipelineLayout graphics_pipeline_layout;
    VkPipeline graphics_pipeline;
    VkPipeline offscreen_graphics_pipeline;
    VkPipeline lights_ui_pipeline;
    VkCommandPool graphics_command_pool;

    Attachment offscreen_position;
    Attachment offscreen_normal;
    Attachment offscreen_albedo;
    Attachment offscreen_depth;

    Attachment object_code;

    VkFramebuffer framebuffers[FRAMES_IN_FLIGHT];
    VkFramebuffer lights_ui_framebuffers[FRAMES_IN_FLIGHT];
    VkFramebuffer offscreen_framebuffer;

    VkDescriptorPool descriptor_pool;
    VkDescriptorPool gbuf_descriptor_pool;
    VkDescriptorPool texture_descriptor_pool;

    Buffer ubo_buffer;
    Buffer deferred_ubo_buffer;

    VkSampler texture_sampler;
    VkSampler gbuf_sampler;

    VkDescriptorSetLayout desc_set_layout;
    VkDescriptorSetLayout gbuf_desc_set_layout;
    VkDescriptorSetLayout texture_set_layout;

    VkDescriptorSet desc_set;
    VkDescriptorSet gbuf_desc_set;

    VkCommandBuffer command_buffer;

    VkFence commands_executed_fence;
    VkSemaphore image_available_semaphore;
    VkSemaphore draw_finished_semaphore;

    size_t current_frame;
} Render;
static Render render;

static void render_swapchain_dependent_init();
static void recreate_swapchain();

void create_window()
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
     
    g_window = glfwCreateWindow(
            mode->width, mode->height, "Demo", monitor, NULL);
    glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(g_window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
}

void create_instance()
{
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = APP_NAME,
        .applicationVersion = VK_MAKE_VERSION(0, 0, 1),
        .pEngineName = ENGINE_NAME,
        .engineVersion = VK_MAKE_VERSION(0, 0, 1),
        .apiVersion = VK_API_VERSION_1_2,
    };
    uint32_t glfw_ext_count;
    const char* *glfw_extensions;
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = glfw_ext_count,
        .ppEnabledExtensionNames = glfw_extensions,
    };
    if (VALIDATION_ENABLED) {
        create_info.enabledLayerCount = 1;
        create_info.ppEnabledLayerNames = (const char *const *) VALIDATION_LAYERS;
    } else {
        create_info.enabledLayerCount = 0;
    }
    if (vkCreateInstance(&create_info, NULL, &render.instance) != VK_SUCCESS) {
        fatal("Failed to create instance");
    }
}

void pick_physical_device()
{
    uint32_t dev_count;
    if (vkEnumeratePhysicalDevices(render.instance, &dev_count, NULL) !=
            VK_SUCCESS) fatal("Failed to enumerate physical devices.");
    VkPhysicalDevice *devices = malloc_nofail(
                                        sizeof(VkPhysicalDevice) * dev_count);
    if (vkEnumeratePhysicalDevices(render.instance, &dev_count, devices) !=
            VK_SUCCESS) fatal("Failed to enumerate physical devices.");

    VkPhysicalDevice result = VK_NULL_HANDLE;
    int graphics;
    int present;
    for (size_t i=0; i < dev_count; i++) {
        VkPhysicalDevice g_device = devices[i];

        // Find graphics and present queue families
        // TODO use a specialized transfer queue
        uint32_t queue_family_count;
        vkGetPhysicalDeviceQueueFamilyProperties(
                g_device, &queue_family_count, NULL);
        VkQueueFamilyProperties *queue_families =
           malloc_nofail(sizeof(VkQueueFamilyProperties) * queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
                                 g_device, &queue_family_count, queue_families);

        graphics = -1;
        for (int j=0; j < queue_family_count; j++) {
            if (queue_families[j].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                graphics = j;
                break;
            }
        } 
        if (graphics == -1)
            continue;

        present = -1;
        for (int j=0; j < queue_family_count; j++) {
            VkBool32 present_support = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(
                                g_device, j, render.surface, &present_support);

            if (present_support == VK_TRUE) {
                present = j;
                break;
            }
        } 
        if (present == -1)
            continue;

        mem_free(queue_families);

        // Check if neccessary extensions are supported
        uint32_t ext_count;
        vkEnumerateDeviceExtensionProperties(g_device, NULL, &ext_count, NULL);
        VkExtensionProperties *available_extensions =
                       malloc_nofail(sizeof(VkExtensionProperties) * ext_count);
        vkEnumerateDeviceExtensionProperties(
                               g_device, NULL, &ext_count, available_extensions);

        bool extensions_supported = true;
        for (int j=0; j < DEVICE_EXTENSION_COUNT; j++) {
            bool found = false;
            for (int k=0; k < ext_count; k++) {
                if (!strcmp(DEVICE_EXTENSIONS[j],
                            available_extensions[k].extensionName)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                extensions_supported = false;
                break;
            }
        }    
        mem_free(available_extensions);

        if (!extensions_supported) continue;

        // Make sure that surface format count isn't 0
        // TODO check for actual needed format
        uint32_t format_count;
        vkGetPhysicalDeviceSurfaceFormatsKHR(
                                g_device, render.surface, &format_count, NULL);
        if (format_count == 0) continue;

        // Make sure that present mode count isn't 0
        // TODO check for actual needed present mode
        uint32_t present_mode_count;
        vkGetPhysicalDeviceSurfacePresentModesKHR(
                            g_device, render.surface, &present_mode_count, NULL);
        if (present_mode_count == 0) continue;

        // Multisampling
        VkPhysicalDeviceFeatures supported_features;
        vkGetPhysicalDeviceFeatures(g_device, &supported_features);
        if (!supported_features.samplerAnisotropy) continue;

        result = g_device;
        break;
    }

    mem_free(devices); 
    if (result == VK_NULL_HANDLE) {
        fatal("Failed to find a suitable physical g_device.");
    } else {
        render.graphics_family = (uint32_t) graphics;
        render.present_family = (uint32_t) present;
        g_physical_device = result;
    }
}

void create_logical_device()
{
    uint32_t queue_count;
    if (render.graphics_family != render.present_family) {
        queue_count = 2;
    } else {
        queue_count = 1;
    }

    VkDeviceQueueCreateInfo* queue_create_infos =
        malloc_nofail(sizeof(*queue_create_infos) * queue_count);

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo graphics_queue_create_info =  {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = render.graphics_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    queue_create_infos[0] = graphics_queue_create_info;
    
    // TODO use a separate queue for presentation (IMPORTANT!)
    if (queue_count > 1) {
        VkDeviceQueueCreateInfo present_queue_create_info = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = render.present_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };
        queue_create_infos[1] = present_queue_create_info;
    }

    VkPhysicalDeviceFeatures features = {
        .samplerAnisotropy = VK_TRUE,
    };

    VkDeviceCreateInfo device_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = queue_count,
        .pQueueCreateInfos = queue_create_infos,
        .pEnabledFeatures = &features,
        .enabledExtensionCount = DEVICE_EXTENSION_COUNT,
        .ppEnabledExtensionNames = (const char *const *) DEVICE_EXTENSIONS,
        .enabledLayerCount = 0,
    };

    if (vkCreateDevice(
            g_physical_device, &device_create_info, NULL, &g_device) !=
            VK_SUCCESS) fatal("Failed to create logical g_device.");

    vkGetDeviceQueue(
        g_device, render.graphics_family, 0, &render.graphics_queue);
    vkGetDeviceQueue(
        g_device, render.present_family, 0, &render.present_queue);
    mem_free(queue_create_infos);
}

void create_command_pool()
{
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = render.graphics_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    if (vkCreateCommandPool(g_device, &pool_info, NULL, 
            &render.graphics_command_pool) != VK_SUCCESS) {
        fatal("Failed to create command pool.");
    }
}

void create_texture_sampler()
{
    VkSamplerCreateInfo texture_sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = 16.0f,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .minLod = 0.0f,
        .maxLod = 1.0,
        .mipLodBias = 0.0f,
    };

    if (vkCreateSampler(g_device, &texture_sampler_info, NULL,
                &render.texture_sampler) != VK_SUCCESS) {
        fatal("Failed to create texture sampler.");
    }
}

void render_init() {
    create_window();
    create_instance();

    if (glfwCreateWindowSurface(
            render.instance, g_window, NULL, &render.surface) != VK_SUCCESS) {
        fatal("Failed to create surface.");
    }

    pick_physical_device();
    create_logical_device();
    create_command_pool();
    create_texture_sampler();

    // CREATE DESCRIPTOR SET LAYOUTS AND PIPELINE LAYOUT
    VkDescriptorSetLayoutBinding mrt_ubo_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    };
    VkDescriptorSetLayoutBinding deferred_ubo_binding = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    VkDescriptorSetLayoutBinding deferred_lights_sbo_binding = {
        .binding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    VkDescriptorSetLayoutBinding desc_set_bindings[3] = {
        mrt_ubo_binding, deferred_ubo_binding, deferred_lights_sbo_binding,
    };
    VkDescriptorSetLayoutCreateInfo desc_set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = desc_set_bindings,
    };
    if (vkCreateDescriptorSetLayout(g_device, &desc_set_info, NULL,
            &render.desc_set_layout) != VK_SUCCESS) {
        fatal("Failed to create descriptor set layout.");
    }

    // CREATE DESCRIPTOR POOLS

    // UBO descriptor pool
    // MRT vertex shader UBO
    // Deferred fragment shader UBO
    VkDescriptorPoolSize ub_pool_size = {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 2,
    };
    VkDescriptorPoolSize sb_pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
    };
    VkDescriptorPoolSize pool_sizes[2] = {
        ub_pool_size, sb_pool_size,
    };
    VkDescriptorPoolCreateInfo desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
        .maxSets = 1,
    };
    if (vkCreateDescriptorPool(
            g_device, &desc_pool_info, NULL, &render.descriptor_pool)
            != VK_SUCCESS) {
        fatal("Failed to create descriptor pool.");
    }

    // G-buffer descriptor pool
    VkDescriptorPoolSize gbuf_desc_pool_size = {
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 3,
    };
    VkDescriptorPoolCreateInfo gbuf_desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 1,
        .pPoolSizes = &gbuf_desc_pool_size,
        .maxSets = 1,
    };
    if (vkCreateDescriptorPool(
            g_device, &gbuf_desc_pool_info, NULL, &render.gbuf_descriptor_pool)
            != VK_SUCCESS) {
        fatal("Failed to create descriptor pool.");
    }

    // Texture descriptor pool
    VkDescriptorPoolSize texture_pool_size = {
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = MAX_TEXTURES,
    };
    VkDescriptorPoolCreateInfo texture_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 1,
        .pPoolSizes = &texture_pool_size,
        .maxSets = MAX_TEXTURES,
    };
    if (vkCreateDescriptorPool(
            g_device, &texture_pool_info, NULL, &render.texture_descriptor_pool)
            != VK_SUCCESS) {
        fatal("Failed to create texture descriptor pool.");
    }

    // Allocate UBO descriptor set
    VkDescriptorSetAllocateInfo desc_set_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = render.descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &render.desc_set_layout,
    };
    if (vkAllocateDescriptorSets(
            g_device, &desc_set_alloc_info, &render.desc_set
            ) != VK_SUCCESS) {
        fatal("Failed to allocate descriptor sets.");
    }

    VkDescriptorSetLayoutBinding position_gbuf_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    VkDescriptorSetLayoutBinding normal_gbuf_binding = {
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    VkDescriptorSetLayoutBinding albedo_gbuf_binding = {
        .binding = 2,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    VkDescriptorSetLayoutBinding gbuf_bindings[3] = {
        position_gbuf_binding, normal_gbuf_binding, albedo_gbuf_binding,
    };
    VkDescriptorSetLayoutCreateInfo gbuf_desc_set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = gbuf_bindings,
    };
    if (vkCreateDescriptorSetLayout(g_device, &gbuf_desc_set_info, NULL,
            &render.gbuf_desc_set_layout) != VK_SUCCESS) {
        fatal("Failed to create descriptor set layout.");
    }

    // Create G-buffer sampler
    VkSamplerCreateInfo gbuf_sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = 1.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
        .unnormalizedCoordinates = VK_FALSE,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .minLod = 0.0f,
        .maxLod = 1.0,
        .mipLodBias = 0.0f,
    };

    if (vkCreateSampler(g_device, &gbuf_sampler_info, NULL,
                &render.gbuf_sampler) != VK_SUCCESS) {
        fatal("Failed to create G-buffer sampler.");
    }

    // Allocate descriptor set for G-buffer attachments
    VkDescriptorSetAllocateInfo gbuf_desc_set_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = render.gbuf_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &render.gbuf_desc_set_layout,
    };
    if (vkAllocateDescriptorSets(
            g_device, &gbuf_desc_set_alloc_info, &render.gbuf_desc_set
            ) != VK_SUCCESS) {
        fatal("Failed to allocate descriptor sets.");
    }

    // Texture sampler descriptor set layout
    VkDescriptorSetLayoutBinding texture_sampler_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    VkDescriptorSetLayoutCreateInfo texture_set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &texture_sampler_binding,
    };
    if (vkCreateDescriptorSetLayout(g_device, &texture_set_info, NULL,
            &render.texture_set_layout) != VK_SUCCESS) {
        fatal("Failed to create texture descriptor set layout.");
    }

    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };

    VkDescriptorSetLayout set_layouts[3] = {
        render.desc_set_layout, render.texture_set_layout,
        render.gbuf_desc_set_layout
    };

    const VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 3,
        .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
    };
    VkPipelineLayout pipeline_layout;
    if (vkCreatePipelineLayout(
                            g_device,
                            &pipeline_layout_info,
                            NULL,
                            &render.graphics_pipeline_layout) != VK_SUCCESS) {
        fatal("Failed to create pipeline layout.");
    }

    // CREATE SYNCHRONIZATION PRIMITIVES
    VkSemaphoreCreateInfo semaphore_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    vkCreateSemaphore(g_device, &semaphore_info, NULL,
            &render.image_available_semaphore);
    vkCreateSemaphore(g_device, &semaphore_info, NULL,
            &render.draw_finished_semaphore);
    vkCreateFence(g_device, &fence_info, NULL,
            &render.commands_executed_fence);

    // CREATE BUFFERS
    // TODO Change MRT UBO to host coherent
    create_buffer(
            sizeof(MrtUbo),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &render.ubo_buffer
    );
    create_buffer(
            sizeof(DeferredUbo),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &render.deferred_ubo_buffer
    );

    // MRT UBO
    VkDescriptorBufferInfo buffer_info = {
        .buffer = render.ubo_buffer.buffer,
        .offset = 0,
        .range = sizeof(MrtUbo),
    };

    VkWriteDescriptorSet uniform_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = render.desc_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &buffer_info,
    };
    vkUpdateDescriptorSets(g_device, 1, &uniform_write, 0, NULL);

    // Deferred UBO
    VkDescriptorBufferInfo deferred_buffer_info = {
        .buffer = render.deferred_ubo_buffer.buffer,
        .offset = 0,
        .range = sizeof(DeferredUbo),
    };

    VkWriteDescriptorSet deferred_uniform_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = render.desc_set,
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &deferred_buffer_info,
    };

    vkUpdateDescriptorSets(g_device, 1, &deferred_uniform_write, 0, NULL);

    render.current_frame = 0;

    // ALLOCATE COMMAND BUFFERS
    VkCommandBufferAllocateInfo cmdbuf_allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = render.graphics_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(g_device, &cmdbuf_allocate_info,
            &render.command_buffer) != VK_SUCCESS) {
        fatal("Failed to allocate command buffer.");
    }

    render_swapchain_dependent_init();
}

static void create_swapchain()
{
    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(
                    g_physical_device, render.surface, &format_count, NULL);
    VkSurfaceFormatKHR *const formats = 
            malloc_nofail(sizeof(VkSurfaceFormatKHR) * format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(
                g_physical_device, render.surface, &format_count, formats);

    VkSurfaceFormatKHR format;
    bool format_found = false;
    if (format_count == 1 && formats->format == VK_FORMAT_UNDEFINED) {
        VkSurfaceFormatKHR deflt = {
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        };
        format = deflt;
        format_found = true;
    } else {
        for (int i=0; i < format_count; i++) {
            if (formats[i].format == VK_FORMAT_B8G8R8A8_SNORM &&
                    formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                format = formats[i];
                format_found = true;
                break;
            }
        }
    }
    // TODO check if we can really use any format
    if (!format_found) format = *formats;
    mem_free(formats);

    // Choose present mode
    uint32_t present_mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(
            g_physical_device, render.surface, &present_mode_count, NULL);
    VkPresentModeKHR* present_modes =
        malloc_nofail(sizeof(VkPresentModeKHR) * present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        g_physical_device, render.surface, &present_mode_count, present_modes);

    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (size_t i=0; i < present_mode_count; i++) {
        if (present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
            present_mode = present_modes[i];
            break;
        } else if (present_modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            present_mode = present_modes[i];
        }
    }
    mem_free(present_modes);

    // Choose swap extent
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                            g_physical_device, render.surface, &capabilities);
    VkExtent2D extent;
    if (capabilities.currentExtent.width != UINT32_MAX) {
        extent = capabilities.currentExtent;
    } else {
        int width;
        int height;
        glfwGetFramebufferSize(g_window, &width, &height);
        VkExtent2D actual_extent = {
            (uint32_t) width,
            (uint32_t) height,
        };
        actual_extent.width = MAX(capabilities.minImageExtent.width,
                                 MIN(capabilities.maxImageExtent.width,
                                     actual_extent.width));

        actual_extent.height = MAX(capabilities.minImageExtent.height,
                                  MIN(capabilities.maxImageExtent.height,
                                      actual_extent.height));
        extent = actual_extent;
    }

    struct VkSwapchainCreateInfoKHR swapchain_create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = render.surface,
        .minImageCount = FRAMES_IN_FLIGHT, // ! hard condition
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .flags = 0,
        .presentMode = present_mode,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
    };
    render.swapchain_format = format.format;
    render.swapchain_extent = extent;

    if (render.graphics_family != render.present_family) {
        uint32_t queue_family_indices[] = {
            render.graphics_family,
            render.present_family,
        };

        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchain_create_info.queueFamilyIndexCount = 2;
        swapchain_create_info.pQueueFamilyIndices = queue_family_indices;
    } else {
        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    if (vkCreateSwapchainKHR(
      g_device, &swapchain_create_info, NULL, &render.swapchain) != VK_SUCCESS) {
            fatal("Failed to create swapchain.");
    }

    uint32_t image_count;
    vkGetSwapchainImagesKHR(g_device, render.swapchain, &image_count, NULL);
    DBASSERT(image_count == FRAMES_IN_FLIGHT);
    vkGetSwapchainImagesKHR(
          g_device, render.swapchain, &image_count, render.swapchain_images);

    for (uint32_t i=0; i < FRAMES_IN_FLIGHT; i++) {
        create_2d_image_view(render.swapchain_images[i],
            render.swapchain_format, VK_IMAGE_ASPECT_COLOR_BIT,
            &render.swapchain_image_views[i]);
    }
}

static void render_swapchain_dependent_init()
{
    create_swapchain();

    // FINAL COMPOSITION RENDER PASS AND FRAMEBUFFER
    struct VkAttachmentDescription color_attachment = {
        .format = render.swapchain_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    enum {attachment_count = 1};
    struct VkAttachmentDescription attachments[attachment_count] = {
        color_attachment,
    };
    struct VkAttachmentReference color_attachment_ref = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
        .pDepthStencilAttachment = NULL,
    };

    struct VkSubpassDependency dependency_start = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
    };
    struct VkSubpassDependency dependency_end = {
        .srcSubpass = 0,
        .dstSubpass = VK_SUBPASS_EXTERNAL,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
    };
    VkSubpassDependency dependencies[2] = {
        dependency_start, dependency_end
    };

    VkRenderPassCreateInfo render_pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = attachment_count,
        .pAttachments = attachments,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 2,
        .pDependencies = dependencies,
    };

    if (vkCreateRenderPass(
           g_device, &render_pass_info, NULL, &render.render_pass) !=
           VK_SUCCESS) fatal("Failed to create render pass.");

    for (int i=0; i < FRAMES_IN_FLIGHT; i++) {
        VkImageView attachments[1] = {
            render.swapchain_image_views[i],
        };
        VkFramebufferCreateInfo framebuffer_info = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render.render_pass,
            .attachmentCount = attachment_count,
            .pAttachments = attachments,
            .width = render.swapchain_extent.width,
            .height = render.swapchain_extent.height,
            .layers = 1,
        };

        if (vkCreateFramebuffer(g_device, &framebuffer_info, NULL,
                &render.framebuffers[i]) != VK_SUCCESS) {
            fatal("Failed to create framebuffer.");
        }
    }

    // OFFSCREEN RENDER PASS AND FRAMEBUFFERS
    struct VkAttachmentDescription position_attachment = {
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    struct VkAttachmentDescription normal_attachment = {
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    struct VkAttachmentDescription albedo_attachment = {
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkFormat depth_format = find_depth_format();
    struct VkAttachmentDescription offscreen_depth_attachment = {
        .format = depth_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    VkAttachmentDescription offscreen_descs[4] = {
        position_attachment, normal_attachment, albedo_attachment,
        offscreen_depth_attachment,
    };
    VkAttachmentReference offscreen_color_refs[3] = {
        {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
    };
    VkAttachmentReference offscreen_depth_ref = {
        3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription offscreen_subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 3,
        .pColorAttachments = offscreen_color_refs,
        .pDepthStencilAttachment = &offscreen_depth_ref,
    };

    VkRenderPassCreateInfo offscreen_render_pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 4,
        .pAttachments = offscreen_descs,
        .subpassCount = 1,
        .pSubpasses = &offscreen_subpass,
        .dependencyCount = 2,
        .pDependencies = dependencies,
    };

    if (vkCreateRenderPass(
           g_device, &offscreen_render_pass_info, NULL,
           &render.offscreen_render_pass) != VK_SUCCESS)
        fatal("Failed to create render pass.");

    create_attachment(&render.offscreen_depth, render.swapchain_extent.width,
            render.swapchain_extent.height, depth_format,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    create_attachment(&render.offscreen_position, render.swapchain_extent.width,
            render.swapchain_extent.height, position_attachment.format,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    create_attachment(&render.offscreen_albedo, render.swapchain_extent.width,
            render.swapchain_extent.height, albedo_attachment.format,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    create_attachment(&render.offscreen_normal, render.swapchain_extent.width,
            render.swapchain_extent.height, normal_attachment.format,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

    // Write G-buffer descriptors
    VkDescriptorImageInfo gbuf_desc_info = {
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .imageView = render.offscreen_position.view,
        .sampler = render.gbuf_sampler,
    };
    VkWriteDescriptorSet gbuf_desc_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = render.gbuf_desc_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .pImageInfo = &gbuf_desc_info,
    };
    vkUpdateDescriptorSets(g_device, 1, &gbuf_desc_write, 0, NULL);

    gbuf_desc_info.imageView = render.offscreen_normal.view;
    gbuf_desc_write.dstBinding = 1;
    vkUpdateDescriptorSets(g_device, 1, &gbuf_desc_write, 0, NULL);

    gbuf_desc_info.imageView = render.offscreen_albedo.view;
    gbuf_desc_write.dstBinding = 2;
    vkUpdateDescriptorSets(g_device, 1, &gbuf_desc_write, 0, NULL);

    // Offscreen framebuffer
    VkImageView offscreen_attachments[4] = {
        render.offscreen_position.view,
        render.offscreen_normal.view,
        render.offscreen_albedo.view,
        render.offscreen_depth.view,
    };
    VkFramebufferCreateInfo offscreen_framebuffer_info = {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = render.offscreen_render_pass,
        .attachmentCount = 4,
        .pAttachments = offscreen_attachments,
        .width = render.swapchain_extent.width,
        .height = render.swapchain_extent.height,
        .layers = 1,
    };

    if (vkCreateFramebuffer(g_device, &offscreen_framebuffer_info, NULL,
            &render.offscreen_framebuffer) != VK_SUCCESS) {
        fatal("Failed to create framebuffer.");
    }

    // Light indicators pass
    {
        struct VkAttachmentDescription color_attachment = {
            .format = render.swapchain_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };
        struct VkAttachmentDescription depth_attachment = {
            .format = depth_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        struct VkAttachmentDescription object_code_attachment = {
            .format = render.swapchain_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        struct VkAttachmentDescription attachments[] = {
            color_attachment, object_code_attachment, depth_attachment,
        };

        struct VkAttachmentReference color_attachment_ref = {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        struct VkAttachmentReference object_code_attachment_ref = {
            .attachment = 1,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        VkAttachmentReference color_refs[2] = {
            color_attachment_ref, object_code_attachment_ref,
        };

        VkAttachmentReference depth_ref = {
            2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        VkSubpassDescription subpass = {
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 2,
            .pColorAttachments = color_refs,
            .pDepthStencilAttachment = &depth_ref,
        };

        VkRenderPassCreateInfo render_pass_info = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 3,
            .pAttachments = attachments,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 2,
            .pDependencies = dependencies,
        };

        if (vkCreateRenderPass(
            g_device, &render_pass_info, NULL, &render.lights_ui_render_pass) !=
            VK_SUCCESS) fatal("Failed to create render pass.");

        create_attachment(&render.object_code, render.swapchain_extent.width,
                render.swapchain_extent.height, render.swapchain_format,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

        for (int i=0; i < FRAMES_IN_FLIGHT; i++) {
            VkImageView attachments[3] = {
                render.swapchain_image_views[i],
                render.object_code.view,
                render.offscreen_depth.view,
            };
            VkFramebufferCreateInfo framebuffer_info = {
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = render.lights_ui_render_pass,
                .attachmentCount = 3,
                .pAttachments = attachments,
                .width = render.swapchain_extent.width,
                .height = render.swapchain_extent.height,
                .layers = 1,
            };
            if (vkCreateFramebuffer(g_device, &framebuffer_info, NULL,
                    &render.lights_ui_framebuffers[i]) != VK_SUCCESS) {
                fatal("Failed to create framebuffer.");
            }
        }
    }

    // CREATE GRAPHICS PIPELINES (offscreen and final)
    struct VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    const VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width  = (float) render.swapchain_extent.width,
        .height = (float) render.swapchain_extent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const VkRect2D scissor = {
        .offset = {0, 0},
        .extent = render.swapchain_extent,
    };

    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .lineWidth = 1.0f,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
    };

    const VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = VK_FALSE,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineColorBlendAttachmentState color_blend_attachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_A_BIT |
                          VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_R_BIT,
        .blendEnable = VK_FALSE,
    };

    VkPipelineColorBlendStateCreateInfo color_blending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
    };

    const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
    };

    // Final composition pipeline
    rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;

    VkShaderModule deferred_vertex_shader = create_shader_module(
                    "./shaders/deferred.vert.spv");
    VkShaderModule deferred_fragment_shader = create_shader_module(
                "./shaders/deferred.frag.spv");

    struct VkPipelineShaderStageCreateInfo vertex_shader_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = deferred_vertex_shader,
        .pName = "main",
    };

    struct VkPipelineShaderStageCreateInfo fragment_shader_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = deferred_fragment_shader,
        .pName = "main",
    };

    enum {shader_stage_count = 2};
    struct VkPipelineShaderStageCreateInfo shader_stages[shader_stage_count] = {
        vertex_shader_stage_info, fragment_shader_stage_info,
    };
    VkPipelineVertexInputStateCreateInfo empty_vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shader_stages,
        .pVertexInputState = &empty_vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &color_blending,
        .pDepthStencilState = &depth_stencil,
        .layout = render.graphics_pipeline_layout,
        .renderPass = render.render_pass,
        .subpass = 0,
    };

    if (vkCreateGraphicsPipelines(
                              g_device,
                              VK_NULL_HANDLE,
                              1,
                              &pipeline_info,
                              NULL,
                              &render.graphics_pipeline) != VK_SUCCESS) {
        fatal("Failed to create graphics pipeline.");
    }

    vkDestroyShaderModule(g_device, deferred_fragment_shader, NULL);
    vkDestroyShaderModule(g_device, deferred_vertex_shader, NULL);

    // G-buffer write pipeline
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;

    shader_stages[0].module = create_shader_module("./shaders/mrt.vert.spv");
    shader_stages[1].module = create_shader_module("./shaders/mrt.frag.spv");

    VkVertexInputBindingDescription vertex_input_binding_description = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    enum { v_attr_desc_count = 3 };
    VkVertexInputAttributeDescription attribute_descriptions[v_attr_desc_count];

    attribute_descriptions[0].binding = 0;
    attribute_descriptions[0].location = 0;
    attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT; 
    attribute_descriptions[0].offset = offsetof(Vertex, position);

    attribute_descriptions[1].binding = 0;
    attribute_descriptions[1].location = 1;
    attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[1].offset = offsetof(Vertex, normal);

    attribute_descriptions[2].binding = 0;
    attribute_descriptions[2].location = 2;
    attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attribute_descriptions[2].offset = offsetof(Vertex, tex_coord);

    struct VkPipelineVertexInputStateCreateInfo vertex_input_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_input_binding_description,
        .vertexAttributeDescriptionCount = v_attr_desc_count,
        .pVertexAttributeDescriptions = attribute_descriptions,
    };

    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.renderPass = render.offscreen_render_pass;

    VkPipelineColorBlendAttachmentState blend_states[3] = {
        color_blend_attachment, color_blend_attachment, color_blend_attachment, 
    };
    color_blending.attachmentCount = 3;
    color_blending.pAttachments = blend_states;

    if (vkCreateGraphicsPipelines(
                          g_device,
                          VK_NULL_HANDLE,
                          1,
                          &pipeline_info,
                          NULL,
                          &render.offscreen_graphics_pipeline) != VK_SUCCESS) {
        fatal("Failed to create graphics pipeline.");
    }

    vkDestroyShaderModule(g_device, shader_stages[0].module, NULL);
    vkDestroyShaderModule(g_device, shader_stages[1].module, NULL);

    // Light indicators pipeline
    struct VkPipelineInputAssemblyStateCreateInfo point_input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    shader_stages[0].module = create_shader_module("./shaders/lights_ui.vert.spv");
    shader_stages[1].module = create_shader_module("./shaders/lights_ui.frag.spv");

    VkVertexInputBindingDescription lights_ui_input_binding_description = {
        .binding = 0,
        .stride = sizeof(Light),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    VkVertexInputAttributeDescription lights_ui_attr_descs[2];
    lights_ui_attr_descs[0].binding = 0;
    lights_ui_attr_descs[0].location = 0;
    lights_ui_attr_descs[0].format = VK_FORMAT_R32G32B32_SFLOAT; 
    lights_ui_attr_descs[0].offset = offsetof(Light, pos);

    lights_ui_attr_descs[1].binding = 0;
    lights_ui_attr_descs[1].location = 1;
    lights_ui_attr_descs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    lights_ui_attr_descs[1].offset = offsetof(Light, color);

    struct VkPipelineVertexInputStateCreateInfo lights_ui_input_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &lights_ui_input_binding_description,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = lights_ui_attr_descs,
    };

    VkPipelineColorBlendAttachmentState lights_ui_blend_states[2] = {
        color_blend_attachment, color_blend_attachment 
    };
    VkPipelineColorBlendStateCreateInfo lights_ui_color_blending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 2,
        .pAttachments = lights_ui_blend_states,
    };

    VkGraphicsPipelineCreateInfo lights_ui_pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shader_stages,
        .pVertexInputState = &lights_ui_input_info,
        .pInputAssemblyState = &point_input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &lights_ui_color_blending,
        .pDepthStencilState = &depth_stencil,
        .layout = render.graphics_pipeline_layout,
        .renderPass = render.lights_ui_render_pass,
        .subpass = 0,
    };

    if (vkCreateGraphicsPipelines(
                              g_device,
                              VK_NULL_HANDLE,
                              1,
                              &lights_ui_pipeline_info,
                              NULL,
                              &render.lights_ui_pipeline) != VK_SUCCESS) {
        fatal("Failed to create graphics pipeline.");
    }

    vkDestroyShaderModule(g_device, shader_stages[0].module, NULL);
    vkDestroyShaderModule(g_device, shader_stages[1].module, NULL);
}

void render_draw_frame(vec3 cam_pos, vec3 cam_dir, vec3 cam_up) {
    size_t current_frame = render.current_frame;

    // Upload MRT UBO
    MrtUbo uniform;
    mat4 proj;
    glm_perspective(0.6,
        render.swapchain_extent.width /
        (float) render.swapchain_extent.height, 0.01, 1000.0, proj);
    proj[1][1] *= -1;
    mat4 view;
    glm_look(cam_pos, cam_dir, cam_up, view);
    glm_mat4_mul(proj, view, uniform.view_proj);
    upload_to_device_local_buffer(
            (void*) &uniform,
            sizeof(uniform),
            &render.ubo_buffer,
            render.graphics_queue,
            render.graphics_command_pool
    );

    // Upload deferred fragment shader UBO
    DeferredUbo defubo;
    memcpy(&defubo.view_pos, cam_pos, sizeof(vec3));
    defubo.light_count = LIGHT_COUNT;
    upload_to_device_local_buffer(
            (void*) &defubo,
            sizeof(defubo),
            &render.deferred_ubo_buffer,
            render.graphics_queue,
            render.graphics_command_pool
    );

    uint32_t image_index;
    VkResult acquire_image_result =
        vkAcquireNextImageKHR(g_device, render.swapchain, UINT64_MAX,
                render.image_available_semaphore, VK_NULL_HANDLE,
                &image_index);

    if (acquire_image_result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_swapchain();
        return;
    } else if (acquire_image_result != VK_SUCCESS &&
            acquire_image_result != VK_SUBOPTIMAL_KHR) {
        fatal("Failed to acquire swapchain image.");
    }

    vkWaitForFences(
            g_device, 1, &render.commands_executed_fence,
            VK_TRUE, UINT64_MAX);
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
    };
    if (vkBeginCommandBuffer(render.command_buffer, &begin_info) !=
            VK_SUCCESS) {
        fatal("Failed to begin recording command buffer.");
    }
    VkClearValue clear_values[4] = {
        { .color = {0.0f, 0.0f, 0.0f, 1.0f} },
        { .color = {0.0f, 0.0f, 0.0f, 1.0f} },
        { .color = {0.0f, 0.0f, 0.0f, 1.0f} },
        { .depthStencil = {1.0f, 0} },
    };
    VkRenderPassBeginInfo render_pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = render.offscreen_render_pass,
        .framebuffer = render.offscreen_framebuffer,
        .renderArea.offset = {0, 0},
        .renderArea.extent = render.swapchain_extent,
        .clearValueCount = 4,
        .pClearValues = clear_values,
    };

    vkCmdBeginRenderPass(render.command_buffer, &render_pass_info,
                             VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(render.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS, render.offscreen_graphics_pipeline);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(render.command_buffer, 0, 1,
            &scene.vertex_buffer.buffer, &offset);
    vkCmdBindIndexBuffer(
            render.command_buffer, scene.index_buffer.buffer, 0,
            VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(render.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS, render.graphics_pipeline_layout,
            0, 1, &render.desc_set, 0, NULL);

    // Draw the nodes
    for (size_t n=0; n < scene.node_count; n++) {
        Mesh* mesh = scene.nodes[n].mesh;
        if (!mesh) continue;

        mat4 transform;
        memcpy(transform, scene.nodes[n].transform, sizeof(transform));
        Node* parent = scene.nodes[n].parent;
        while (parent) {
            mat4 parent_transform;
            memcpy(parent_transform, parent->transform, sizeof(mat4));
            glm_mat4_mul(parent_transform, transform, transform);
            parent = parent->parent;
        }

        vkCmdPushConstants(
                render.command_buffer,
                render.graphics_pipeline_layout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                sizeof(PushConstants),
                &transform);

        for (size_t p=0; p < mesh->primitives_count; p++) {
            Primitive* primitive = &mesh->primitives[p];
            vkCmdBindDescriptorSets(
                render.command_buffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS, render.graphics_pipeline_layout,
                1, 1, &primitive->texture, 0, NULL);
            vkCmdDrawIndexed(render.command_buffer,
                primitive->index_count, 1, primitive->index_offset,
                primitive->vertex_offset, 0);
        }
    }

    vkCmdEndRenderPass(render.command_buffer);

    render_pass_info.renderPass = render.render_pass;
    render_pass_info.framebuffer = render.framebuffers[current_frame];
    VkClearValue deferred_clear_values[2] = {
        { .color = {0.0f, 0.0f, 0.0f, 0.0f} },
        { .depthStencil = {1.0f, 0} },
    };
    render_pass_info.clearValueCount = 2;
    render_pass_info.pClearValues = deferred_clear_values;

    vkCmdBeginRenderPass(render.command_buffer, &render_pass_info,
                         VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindDescriptorSets(render.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS, render.graphics_pipeline_layout,
            0, 1, &render.desc_set, 0, NULL);
    vkCmdBindDescriptorSets(render.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS, render.graphics_pipeline_layout,
            2, 1, &render.gbuf_desc_set, 0, NULL);
    vkCmdBindPipeline(render.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS, render.graphics_pipeline);
    vkCmdDraw(render.command_buffer, 3, 1, 0, 0);
    vkCmdEndRenderPass(render.command_buffer);

    render_pass_info.renderPass = render.lights_ui_render_pass;
    render_pass_info.framebuffer = render.lights_ui_framebuffers[current_frame];
    render_pass_info.clearValueCount = 2;
    VkClearValue lights_ui_clear_values[2] = {
        { .color = {0.0f, 0.0f, 0.0f, 0.0f} },
        { .color = {0.0f, 0.0f, 0.0f, 0.0f} },
    };
    render_pass_info.pClearValues = lights_ui_clear_values;
    vkCmdBeginRenderPass(render.command_buffer, &render_pass_info,
            VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindVertexBuffers(render.command_buffer, 0, 1,
            &scene.lights_buffer.buffer, &offset);
    vkCmdBindDescriptorSets(render.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS, render.graphics_pipeline_layout,
            0, 1, &render.desc_set, 0, NULL);
    vkCmdBindPipeline(render.command_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS, render.lights_ui_pipeline);
    vkCmdDraw(render.command_buffer, LIGHT_COUNT, 1, 0, 0);
    vkCmdEndRenderPass(render.command_buffer);

    if (vkEndCommandBuffer(render.command_buffer) != VK_SUCCESS) {
        fatal("Failed to record command buffer.");
    }

    VkPipelineStageFlags wait_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &render.image_available_semaphore,
        .pWaitDstStageMask = &wait_mask,
        .commandBufferCount = 1,
        .pCommandBuffers = &render.command_buffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &render.draw_finished_semaphore,
    };

    vkResetFences(g_device, 1, &render.commands_executed_fence);

    if (vkQueueSubmit(render.graphics_queue, 1, &submit_info,
            render.commands_executed_fence) != VK_SUCCESS) {
        fatal("Failed to submit draw command buffer.");
    }

    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &render.draw_finished_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &render.swapchain,
        .pImageIndices = &image_index,
    };
    VkResult present_result = vkQueuePresentKHR(render.present_queue, &present_info);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
            present_result == VK_SUBOPTIMAL_KHR) {
        recreate_swapchain();
    } else if (present_result != VK_SUCCESS) {
        fatal("Failed to present swapchain image.");
    }

    render.current_frame = (current_frame + 1) % 2;

    glfwPollEvents();
}

bool render_exit() {
    return glfwWindowShouldClose(g_window);
}

static void load_texture(void* buffer, size_t len, Texture* texture)
{
    // Load pixels
    int tex_width, tex_height, tex_channels;
    stbi_uc* pixels = stbi_load_from_memory(
        buffer, len, &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
    if (!pixels) fatal("Failed to load texture.");
    uint32_t image_size = tex_width * tex_height * 4;

    // Upload pixels to staging buffer
    Buffer texture_staging = upload_data_to_staging_buffer(pixels, image_size);

    stbi_image_free(pixels);

    // Create texture image
    create_2d_image(tex_width, tex_height,
            VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &texture->image, &texture->memory);

    {
    // Transition image layout for upload
    VkCommandBuffer cmdbuf = begin_one_time_command_buffer(
            render.graphics_command_pool);
    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = texture->image,
        .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .subresourceRange.baseMipLevel = 0,
        .subresourceRange.levelCount = 1,
        .subresourceRange.baseArrayLayer = 0,
        .subresourceRange.layerCount = 1,
    };
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);
    submit_one_time_command_buffer(
            render.graphics_queue, cmdbuf,
            render.graphics_command_pool);
    }

    {
    // Copy staging buffer to image
    VkCommandBuffer cmdbuf = begin_one_time_command_buffer(
            render.graphics_command_pool);
    VkBufferImageCopy region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .imageSubresource.mipLevel = 0,
        .imageSubresource.baseArrayLayer = 0,
        .imageSubresource.layerCount = 1,
        .imageOffset = {0, 0, 0},
        .imageExtent = {
            tex_width,
            tex_height,
            1
        },
    };
    vkCmdCopyBufferToImage(cmdbuf, texture_staging.buffer, texture->image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    submit_one_time_command_buffer(render.graphics_queue, cmdbuf,
            render.graphics_command_pool);

    destroy_buffer(&texture_staging);
    }
    {
    // Transition image layout for shader access
    VkCommandBuffer cmdbuf = begin_one_time_command_buffer(
            render.graphics_command_pool);
    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = texture->image,
        .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .subresourceRange.baseMipLevel = 0,
        .subresourceRange.levelCount = 1,
        .subresourceRange.baseArrayLayer = 0,
        .subresourceRange.layerCount = 1,
    };
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1,
            &barrier);
    submit_one_time_command_buffer(
            render.graphics_queue, cmdbuf,
            render.graphics_command_pool);
    }

    create_2d_image_view(texture->image, VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_ASPECT_COLOR_BIT, &texture->view);

    VkDescriptorSetAllocateInfo tex_set_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = render.texture_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &render.texture_set_layout,
    };
    if (vkAllocateDescriptorSets(
            g_device, &tex_set_alloc_info, &texture->desc_set) != VK_SUCCESS) {
        fatal("Failed to allocate descriptor sets.");
    }

    VkDescriptorImageInfo texture_info = {
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .imageView = texture->view,
        .sampler = render.texture_sampler,
    };
    VkWriteDescriptorSet texture_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = texture->desc_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .pImageInfo = &texture_info,
    };
    vkUpdateDescriptorSets(g_device, 1, &texture_write, 0, NULL);
}

void load_scene()
{
    // LOAD GLTF
    cgltf_options gltf_options = {0};
    cgltf_data* gltf_data = NULL;
    cgltf_result gltf_result = cgltf_parse_file(
                            &gltf_options, "res/cube.glb", &gltf_data);
    if (gltf_result != cgltf_result_success) fatal("Failed to load GLTF.");
    gltf_result = cgltf_load_buffers(&gltf_options, gltf_data, "res/cube.glb");
    if (gltf_result != cgltf_result_success) fatal("Failed to load GLTF buffers.");
    
    // Load materials
    scene.texture_count = gltf_data->materials_count;
    DBASSERT(scene.texture_count <= MAX_TEXTURES);
    scene.textures = malloc_nofail(sizeof(Texture) * scene.texture_count);
    for (size_t i=0; i < scene.texture_count; i++) {
        cgltf_material* gltf_material = &gltf_data->materials[i];
        DBASSERT(gltf_material->has_pbr_metallic_roughness);
        cgltf_texture_view* gltf_texture_view = 
            &gltf_material->pbr_metallic_roughness.base_color_texture;
        cgltf_texture* gltf_texture = gltf_texture_view->texture;
        cgltf_image* gltf_image = gltf_texture->image;
        DBASSERT(!strcmp(gltf_image->mime_type, "image/jpeg"));
        cgltf_buffer_view* image_buffer_view = gltf_image->buffer_view;
        cgltf_buffer* image_buffer = image_buffer_view->buffer;
        void* image_data = image_buffer->data + image_buffer_view->offset;
        size_t image_size = image_buffer_view->size;

        load_texture(image_data, image_size, &scene.textures[i]);
    }

    scene.meshes = malloc_nofail(sizeof(Mesh) * gltf_data->meshes_count);
    scene.mesh_count = gltf_data->meshes_count;

    // Precalculate index and vertex buffer sizes
    size_t index_count = 0;
    size_t vertex_count = 0;
    for (size_t i=0; i < gltf_data->meshes_count; i++) {
        cgltf_mesh* gltf_mesh = &gltf_data->meshes[i];
        for (size_t p=0; p < gltf_mesh->primitives_count; p++) {
            cgltf_primitive* gltf_primitive = &gltf_mesh->primitives[p];
            DBASSERT(gltf_primitive->type == cgltf_primitive_type_triangles);

            for (size_t a=0; a < gltf_primitive->attributes_count; a++) {
                cgltf_attribute* attribute = &gltf_primitive->attributes[a];
                cgltf_accessor* accessor = attribute->data;
                if (attribute->type == cgltf_attribute_type_position) {
                    vertex_count += accessor->count;
                }
            }
            DBASSERT(gltf_primitive->indices->component_type ==
                    cgltf_component_type_r_16u);
            index_count += gltf_primitive->indices->count;
        }
    }
    Vertex* vertices = malloc_nofail(vertex_count * sizeof(Vertex));
    uint16_t* indices = malloc_nofail(index_count * sizeof(uint16_t));

    // Load meshes
    size_t index_offset = 0;
    size_t vertex_offset = 0;
    for (size_t i=0; i < gltf_data->meshes_count; i++) {
        cgltf_mesh* gltf_mesh = &gltf_data->meshes[i];
        Mesh* mesh = &scene.meshes[i];
        mesh->primitives_count = gltf_mesh->primitives_count;
        mesh->primitives = malloc_nofail(
                        sizeof(Primitive) * mesh->primitives_count);
        // Primitives
        for (size_t p=0; p < gltf_mesh->primitives_count; p++) {
            cgltf_primitive* gltf_primitive = &gltf_mesh->primitives[p];

            // Material
            size_t material_index = (size_t)(((char*)gltf_primitive->material -
                    (char*) gltf_data->materials) / sizeof(cgltf_material));
            mesh->primitives[p].texture = scene.textures[material_index].desc_set;

            // Vertices
            mesh->primitives[p].vertex_offset = vertex_offset;
            size_t primitive_vertex_count;
            for (size_t a=0; a < gltf_primitive->attributes_count; a++) {
                cgltf_attribute* attribute = &gltf_primitive->attributes[a];
                cgltf_accessor* accessor = attribute->data;
                size_t count = accessor->count;
                size_t stride = accessor->stride;
                cgltf_buffer_view* buffer_view = accessor->buffer_view;
                cgltf_buffer* buffer = buffer_view->buffer;
                char* data = (char*) buffer->data +
                            buffer_view->offset + accessor->offset;
                
                if (attribute->type == cgltf_attribute_type_position) {
                    for (size_t v=0; v < count; v++) {
                        vec3* pos = (vec3*) data;
                        vertices[vertex_offset + v].position[0] = (*pos)[0];
                        vertices[vertex_offset + v].position[1] = (*pos)[1];
                        vertices[vertex_offset + v].position[2] = (*pos)[2];
                        data += stride;
                    }
                    primitive_vertex_count = count;
                }
                
                if (attribute->type == cgltf_attribute_type_normal) {
                    for (size_t v=0; v < count; v++) {
                        vec3* normal = (vec3*) data;
                        vertices[vertex_offset + v].normal[0] = (*normal)[0];
                        vertices[vertex_offset + v].normal[1] = (*normal)[1];
                        vertices[vertex_offset + v].normal[2] = (*normal)[2];
                        data += stride;
                    }
                }
                
                if (attribute->type == cgltf_attribute_type_texcoord) {
                    for (size_t v=0; v < count; v++) {
                        vec2* texcoord = (vec2*) data;
                        vertices[vertex_offset + v].tex_coord[0] = (*texcoord)[0];
                        vertices[vertex_offset + v].tex_coord[1] = (*texcoord)[1];
                        data += stride;
                    }
                }
            }
            vertex_offset += primitive_vertex_count;

            // Indices
            mesh->primitives[p].index_offset = index_offset;
            cgltf_accessor* index_accessor = gltf_primitive->indices;
            size_t count = index_accessor->count;
            size_t stride = index_accessor->stride;
            cgltf_buffer_view* buffer_view = index_accessor->buffer_view;
            cgltf_buffer* buffer = buffer_view->buffer;
            char* data = (char*) buffer->data +
                        buffer_view->offset + index_accessor->offset;
            
            for (size_t i=0; i < count; i++) {
                uint16_t* index = (uint16_t*) data;
                indices[index_offset + i] = *index;
                data += stride;
            }
            mesh->primitives[p].index_count = count;
            index_offset += count;
        }
    }
    
    // Load nodes
    cgltf_node* gltf_nodes = gltf_data->nodes;
    scene.node_count = gltf_data->nodes_count;
    scene.nodes = malloc_nofail(sizeof(Node) * scene.node_count);

    for (size_t n=0; n < scene.node_count; n++) {
        Node* node = &scene.nodes[n];
        cgltf_node* gltf_node = &gltf_nodes[n];

        mat4 transform = GLM_MAT4_IDENTITY_INIT;
        DBASSERT(!gltf_node->has_matrix);
        if (gltf_node->has_translation) {
            glm_translate(transform, gltf_node->translation);
        }
        if (gltf_node->has_rotation) {
            versor quat;
            memcpy(quat, gltf_node->rotation, sizeof(vec4));
            glm_quat_rotate(transform, quat, transform);
        }
        if (gltf_node->has_scale) glm_scale(transform, gltf_node->scale);

        memcpy(node->transform, transform, sizeof(mat4));

        node->mesh = NULL;
        if (gltf_node->mesh) {
            size_t mesh_index = (size_t) (((char*) gltf_node->mesh -
                        (char*) gltf_data->meshes) / sizeof(cgltf_mesh));
            node->mesh = &scene.meshes[mesh_index];
        }

        node->parent = NULL;
        if (gltf_node->parent) {
            size_t parent_index = (size_t) (((char*) gltf_node->parent -
                        (char*) gltf_nodes) / sizeof(cgltf_node));
            node->parent = &scene.nodes[parent_index];
        }

        node->children_count = gltf_node->children_count;
        node->children = malloc_nofail(sizeof(Node*) * node->children_count);
        for (size_t c=0; c < gltf_node->children_count; c++) {
            size_t child_index = (size_t) (((char*) gltf_node->children[c] -
                        (char*) gltf_data->nodes) / sizeof(cgltf_node));
            node->children[c] = &scene.nodes[child_index];    
        }
    }

    device_local_buffer_from_data(
            (void*) vertices,
            sizeof(Vertex) * vertex_count,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            render.graphics_queue,
            render.graphics_command_pool,
            &scene.vertex_buffer
    );
    device_local_buffer_from_data(
            (void*) indices,
            sizeof(uint16_t) * index_count,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            render.graphics_queue,
            render.graphics_command_pool,
            &scene.index_buffer
    );
    mem_free(vertices);
    mem_free(indices);

    // Load lights
    Light light1 = {
        .pos = { 3.0, 6.0, 12.0 },
        .color = { 0.0, 0.0, 1.0 },
    };
    Light light2 = {
        .pos = { -8.0, -2.0, 8.0 },
        .color = { 1.0, 1.0, 0.0 },
    };
    Light lights[2] = {light1, light2};

    scene.light_count = LIGHT_COUNT;
    scene.lights = mem_alloc(sizeof(Light) * scene.light_count);
    memcpy(scene.lights, lights, sizeof(Light) * scene.light_count);

    create_buffer(
            sizeof(Light) * LIGHT_COUNT,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &scene.lights_buffer
    );
    upload_to_device_local_buffer(
            (void*) lights,
            sizeof(Light) * LIGHT_COUNT,
            &scene.lights_buffer,
            render.graphics_queue,
            render.graphics_command_pool
    );

    // Deferred lights SBO
    VkDescriptorBufferInfo lights_sbo_info = {
        .buffer = scene.lights_buffer.buffer,
        .offset = 0,
        .range = sizeof(Light) * LIGHT_COUNT,
    };

    VkWriteDescriptorSet lights_sbo_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = render.desc_set,
        .dstBinding = 2,
        .dstArrayElement = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &lights_sbo_info,
    };
    vkUpdateDescriptorSets(g_device, 1, &lights_sbo_write, 0, NULL);

    cgltf_free(gltf_data);
}

static void cleanup_swapchain()
{
    vkDestroyDescriptorPool(g_device, render.descriptor_pool, NULL);
    vkDestroyDescriptorPool(g_device, render.gbuf_descriptor_pool, NULL);

    vkDestroySemaphore(
            g_device, render.image_available_semaphore, NULL);
    vkDestroySemaphore(
            g_device, render.draw_finished_semaphore, NULL);
    vkDestroyFence(g_device, render.commands_executed_fence, NULL);

    for (size_t i=0; i < FRAMES_IN_FLIGHT; i++) {
        vkDestroyFramebuffer(g_device, render.framebuffers[i], NULL);
        vkDestroyFramebuffer(g_device, render.lights_ui_framebuffers[i], NULL);
    }
    vkDestroyFramebuffer(g_device, render.offscreen_framebuffer, NULL);
    
    destroy_attachment(&render.offscreen_position);
    destroy_attachment(&render.offscreen_normal);
    destroy_attachment(&render.offscreen_albedo);
    destroy_attachment(&render.offscreen_depth);
    destroy_attachment(&render.object_code);

    vkDestroyPipeline(g_device, render.graphics_pipeline, NULL);
    vkDestroyPipeline(g_device, render.offscreen_graphics_pipeline, NULL);
    vkDestroyPipeline(g_device, render.lights_ui_pipeline, NULL);
    vkDestroyPipelineLayout(g_device, render.graphics_pipeline_layout, NULL);

    vkDestroyRenderPass(g_device, render.render_pass, NULL);
    vkDestroyRenderPass(g_device, render.offscreen_render_pass, NULL);
    vkDestroyRenderPass(g_device, render.lights_ui_render_pass, NULL);

    for (uint32_t i=0; i < FRAMES_IN_FLIGHT; i++) {
        vkDestroyImageView(g_device, render.swapchain_image_views[i], NULL);
    }

    vkDestroySwapchainKHR(g_device, render.swapchain, NULL);
}

void render_destroy()
{
    vkDeviceWaitIdle(g_device);

    cleanup_swapchain();
    destroy_scene(&scene);

    destroy_buffer(&render.ubo_buffer);
    destroy_buffer(&render.deferred_ubo_buffer);

    vkDestroySampler(g_device, render.texture_sampler, NULL);
    vkDestroySampler(g_device, render.gbuf_sampler, NULL);

    vkDestroyDescriptorPool(g_device, render.texture_descriptor_pool, NULL);

    vkDestroyDescriptorSetLayout(g_device, render.desc_set_layout, NULL);
    vkDestroyDescriptorSetLayout(g_device, render.gbuf_desc_set_layout, NULL);
    vkDestroyDescriptorSetLayout(g_device, render.texture_set_layout, NULL);

    vkDestroyCommandPool(g_device, render.graphics_command_pool, NULL);
    vkDestroyDevice(g_device, NULL);
    vkDestroySurfaceKHR(render.instance, render.surface, NULL);
    vkDestroyInstance(render.instance, NULL);

    glfwDestroyWindow(g_window);
    glfwTerminate();
}

static void create_2d_image(uint32_t width, uint32_t height,
        VkSampleCountFlagBits samples, VkFormat format, VkImageTiling tiling,
        VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
        VkImage* image, VkDeviceMemory* memory)
{
    VkImageCreateInfo image_create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .extent.width = width,
        .extent.height = height,
        .extent.depth = 1,
        .mipLevels = 1,
        .arrayLayers = 1,
        .format = format,
        .tiling = tiling,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .samples = samples,
    };

    if (vkCreateImage(g_device, &image_create_info, NULL, image) != VK_SUCCESS) {
        fatal("Failed to create image.");
    }

    VkMemoryRequirements memory_requirements;
    vkGetImageMemoryRequirements(g_device, *image, &memory_requirements);

    int memory_type_index = find_memory_type(
            memory_requirements,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    if (memory_type_index < 0) {
        fatal("Failed to find suitable memory type for image creation.");
    }

    VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex = (uint32_t) memory_type_index,
    };

    if (vkAllocateMemory(g_device, &allocate_info, NULL, memory)
            != VK_SUCCESS) fatal("Failed to allocate image memory.");

    vkBindImageMemory(g_device, *image, *memory, 0);
}

static void create_2d_image_view(VkImage image, VkFormat format,
        VkImageAspectFlags aspect_flags, VkImageView* image_view)
{
    VkImageViewCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components.r = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.g = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.b = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.a = VK_COMPONENT_SWIZZLE_IDENTITY,
        .subresourceRange.aspectMask = aspect_flags,
        .subresourceRange.baseMipLevel = 0,
        .subresourceRange.levelCount = 1,
        .subresourceRange.baseArrayLayer = 0,
        .subresourceRange.layerCount = 1,
    };
    if (vkCreateImageView(g_device, &create_info, NULL, image_view) != VK_SUCCESS) {
        fatal("Failed to create image views.");
    }
}

static VkFormat find_depth_format() {
    enum {candidate_count = 3};
    const VkFormat const candidates[candidate_count] = {
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
    };

    VkFormat depth_format;
    bool found = false;
    for (size_t i=0; i < candidate_count; i++) {
        VkFormatProperties properties;
        vkGetPhysicalDeviceFormatProperties(
                g_physical_device, candidates[i], &properties);
        if (properties.optimalTilingFeatures &
                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            depth_format = candidates[i];
            found = true;
            break;
        }
    }
    if (!found) fatal("Failed to find depth format.");
    return depth_format;
}

static VkShaderModule create_shader_module(const char* path)
{
    char *shader_code; 
    size_t shader_code_size; 
    if (read_binary_file(path, &shader_code, &shader_code_size)) {
        fatal("Failed to read vertex shader");
    }

    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shader_code_size,
        .pCode = (const uint32_t*) shader_code,
    };

    VkShaderModule shader_module;
    if (vkCreateShaderModule(
            g_device, &create_info, NULL, &shader_module) !=
            VK_SUCCESS) {
        fatal("Failed to create shader module.");
    }

    mem_free(shader_code);
    return shader_module;
}

static VkCommandBuffer begin_one_time_command_buffer(VkCommandPool command_pool)
{
    VkCommandBufferAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandPool = command_pool,
        .commandBufferCount = 1,
    };

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(g_device, &allocate_info, &command_buffer);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkBeginCommandBuffer(command_buffer, &begin_info);

    return command_buffer;
}

static void submit_one_time_command_buffer(VkQueue queue,
        VkCommandBuffer command_buffer, VkCommandPool command_pool)
{
    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };

    vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);

    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(g_device, command_pool, 1, &command_buffer);
}

static int find_memory_type(
        VkMemoryRequirements memory_requirements,
        VkMemoryPropertyFlags required_properties
) { 
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(g_physical_device, &memory_properties);
    int memory_type_index = -1;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        if ((memory_requirements.memoryTypeBits & (1 << i)) &&
                (memory_properties.memoryTypes[i].propertyFlags &
                 required_properties) == required_properties) {
            memory_type_index = (int) i;
            break;
        }
    }
    return memory_type_index;
}

// TODO straighten out shady return values
static int create_buffer(
        size_t size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, Buffer* buffer)
{
    VkDeviceSize device_size = size;
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = device_size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    if (vkCreateBuffer(g_device, &buffer_info, NULL, &buffer->buffer) != VK_SUCCESS) {
        return 1;
    }

    VkMemoryRequirements memory_requirements;
    vkGetBufferMemoryRequirements(g_device, buffer->buffer, &memory_requirements);

    VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex = find_memory_type(memory_requirements, properties),
    };

    if (vkAllocateMemory(g_device, &allocate_info, NULL, &buffer->memory)
            != VK_SUCCESS) return 2;

    vkBindBufferMemory(g_device, buffer->buffer, buffer->memory, 0);
    return 0;
}

static void copy_buffer(
        VkQueue queue,
        VkCommandPool command_pool,
        VkBuffer src_buffer, VkBuffer dst_buffer,
        VkDeviceSize device_size
) {
    VkCommandBuffer command_buffer = begin_one_time_command_buffer(command_pool);

    VkBufferCopy copy_region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = device_size,
    };
    vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);
    submit_one_time_command_buffer(queue, command_buffer, command_pool);
}

static Buffer upload_data_to_staging_buffer(void* data, size_t size)
{
    VkDeviceSize device_size = size;

    Buffer staging_buffer;
    create_buffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            // TODO use non-coherent memory with a flush
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &staging_buffer
    );

    void *staging_buffer_mapped;
    vkMapMemory(
            g_device, staging_buffer.memory, 0, device_size, 0,
            &staging_buffer_mapped);
    memcpy(staging_buffer_mapped, data, size);
    vkUnmapMemory(g_device, staging_buffer.memory);
    return staging_buffer;
}

static void upload_to_device_local_buffer(
        void* data,
        size_t size,
        Buffer* destination,
        VkQueue queue,
        VkCommandPool command_pool)
{
    Buffer staging_buffer = upload_data_to_staging_buffer(data, size);
    copy_buffer(
        queue, command_pool, staging_buffer.buffer, destination->buffer, size);
    destroy_buffer(&staging_buffer);
}

static void device_local_buffer_from_data(
        void* data,
        size_t size,
        VkBufferUsageFlags usage,
        VkQueue queue,
        VkCommandPool command_pool,
        Buffer* buffer)
{
    create_buffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            buffer);
    upload_to_device_local_buffer(
            data,
            size,
            buffer,
            queue,
            command_pool);
}

static void recreate_swapchain()
{
    int width = 0, height = 0;
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(g_window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(g_device);

    cleanup_swapchain();
    render_swapchain_dependent_init();
}
