#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <cglm/cglm.h>

#include "utils.h"
#include "alloc.h"

#include "render.h"

#define APP_NAME "Demo"
#define ENGINE_NAME "None"

typedef struct Uniform {
    mat4 view_proj;
} Uniform;

typedef struct PushConstants {
    mat4 model;
} PushConstants;

static VkInstance create_instance();
static VkPhysicalDevice pick_physical_device(
        const VkInstance instance, const VkSurfaceKHR surface,
        uint32_t *const o_present_family, uint32_t *const o_graphics_family);
static VkDevice create_logical_device(
        VkPhysicalDevice physical_device, VkSurfaceKHR surface,
        uint32_t graphics_family, uint32_t present_family,
        VkQueue* o_graphics_queue, VkQueue* o_present_queue);
static VkSwapchainKHR create_swapchain(const VkPhysicalDevice physical_device,
        const VkDevice device,
        const VkSurfaceKHR surface, uint32_t graphics_family,
        uint32_t present_family, GLFWwindow* const window,
        VkFormat* const o_swapchain_format,
        VkExtent2D* const o_swapchain_extent,
        VkImage* *const o_images);
static VkImageView* create_swapchain_image_views(VkDevice device, VkFormat format,
        VkImage* images);
static VkRenderPass create_render_pass(VkFormat swapchain_format,
        VkPhysicalDevice physical_device, VkDevice device);
static VkDescriptorSetLayout create_descriptor_set_layout(VkDevice device);
static VkPipeline create_graphics_pipeline(
        VkDevice device,
        VkExtent2D swapchain_extent,
        VkSampleCountFlagBits sample_count,
        VkDescriptorSetLayout descriptor_set_layout,
        VkRenderPass render_pass,
        VkPipelineLayout* o_layout);
static VkImage create_color_image(
        VkDevice device, VkPhysicalDevice physical_device, VkFormat format,
        VkExtent2D extent, VkSampleCountFlagBits sample_count,
        VkCommandPool command_pool, VkQueue queue, VkImageView* o_view,
        VkDeviceMemory* o_memory);
static VkImage create_depth_image(
        VkDevice device, VkPhysicalDevice physical_device,
        VkExtent2D extent, VkSampleCountFlagBits sample_count,
        VkCommandPool command_pool, VkQueue queue, VkImageView* o_view,
        VkDeviceMemory* o_memory);
static VkFramebuffer* create_framebuffers(
        VkDevice device,
        VkImageView color_image_view, VkImageView depth_image_view,
        VkImageView* swapchain_image_views, VkRenderPass render_pass,
        VkExtent2D swapchain_extent);
static VkDescriptorPool create_descriptor_pool(
        VkDevice device, VkDescriptorType* descriptor_types,
        size_t descriptor_type_count
);
static int find_memory_type(
        VkPhysicalDevice physical_device,
        VkMemoryRequirements memory_requirements,
        VkMemoryPropertyFlags required_properties
);
static VkBuffer device_local_buffer_from_data(
        void* data,
        size_t size,
        VkBufferUsageFlags usage,
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkQueue queue,
        VkCommandPool command_pool,
        VkDeviceMemory *o_memory
);
VkBuffer* create_uniform_buffers(
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkDeviceMemory* *const o_memories
);
static VkDescriptorSet* create_descriptor_sets(
        VkDevice device,
        VkDescriptorPool descriptor_pool,
        VkDescriptorSetLayout descriptor_set_layout,
        VkBuffer* uniform_buffers
);
static void upload_to_device_local_buffer(
        void* data,
        size_t size,
        VkBuffer destination,
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkQueue queue,
        VkCommandPool command_pool
);
static VkCommandBuffer* record_command_buffers(
        VkDevice device,
        VkCommandPool command_pool,
        VkRenderPass render_pass,
        VkFramebuffer* swapchain_framebuffers,
        VkExtent2D swapchain_extent,
        VkPipelineLayout graphics_pipeline_layout,
        VkPipeline graphics_pipeline,
        VkBuffer vertex_buffer,
        VkBuffer index_buffer,
        uint32_t index_count,
        VkDescriptorSet* descriptor_sets,
        PushConstants* push_constants
);

enum { VALIDATION_ENABLED = 1 };

const char *const VALIDATION_LAYERS[] = {
    "VK_LAYER_KHRONOS_validation"
};

#define DEVICE_EXTENSION_COUNT 1
const char *const DEVICE_EXTENSIONS[DEVICE_EXTENSION_COUNT] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// TODO account for no multisampling support on the device
#define SAMPLE_COUNT 2

#define VERTEX_SHADER_PATH "./shaders/vert.spv"
#define FRAGMENT_SHADER_PATH "./shaders/frag.spv"

typedef struct Render {
    GLFWwindow* window;
    VkInstance instance;
    VkSurfaceKHR surface;
    VkPhysicalDevice physical_device;
    uint32_t graphics_family;
    uint32_t present_family;
    VkDevice device;
    VkQueue graphics_queue;
    VkQueue present_queue;
    VkSwapchainKHR swapchain;
    VkFormat swapchain_format;
    VkExtent2D swapchain_extent;
    VkImage* swapchain_images;
    VkImageView* swapchain_image_views;
    VkRenderPass render_pass;
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout graphics_pipeline_layout;
    VkPipeline graphics_pipeline;
    VkCommandPool graphics_command_pool;

    VkImage color_image;
    VkImageView color_image_view;
    VkDeviceMemory color_image_memory;

    VkImage depth_image;
    VkImageView depth_image_view;
    VkDeviceMemory depth_image_memory;

    VkFramebuffer* framebuffers;

    VkDescriptorPool descriptor_pool;

    VkBuffer vertex_buffer;
    VkDeviceMemory vertex_buffer_memory;

    VkBuffer index_buffer;
    VkDeviceMemory index_buffer_memory;

    VkBuffer* uniform_buffers;
    VkDeviceMemory* uniform_buffers_memories;

    VkDescriptorSet* descriptor_sets;

    VkCommandBuffer* command_buffers;

    VkFence commands_executed_fences[2];
    VkSemaphore image_available_semaphores[2];
    VkSemaphore draw_finished_semaphores[2];

    size_t current_frame;
} Render;

static void render_swapchain_dependent_init(Render* self, GLFWwindow* window);
static void recreate_swapchain(Render* self, GLFWwindow* window);

Render* render_init() {
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
     
    GLFWwindow* window = glfwCreateWindow(
            mode->width, mode->height, "Demo", monitor, NULL);

    Render* self = (Render*) malloc_nofail(sizeof(Render));
    self->window = window;
    self->instance = create_instance();

    if (glfwCreateWindowSurface(
            self->instance, window, NULL, &self->surface) != VK_SUCCESS) {
        fatal("Failed to create surface.");
    }

    self->physical_device = pick_physical_device(
            self->instance, self->surface, &self->graphics_family,
            &self->present_family);
    self->device = create_logical_device(self->physical_device, self->surface,
            self->graphics_family, self->present_family, &self->graphics_queue,
            &self->present_queue);

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = self->graphics_family,
    };
    if (vkCreateCommandPool(self->device, &pool_info, NULL, 
            &self->graphics_command_pool) != VK_SUCCESS) {
        fatal("Failed to create command pool.");
    }

    self->vertex_buffer = VK_NULL_HANDLE;
    self->index_buffer = VK_NULL_HANDLE;

    // TODO merge descriptor set layout and pipeline layout creation
    VkDescriptorSetLayoutBinding uniform_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    };
    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &uniform_binding,
    };
    if (vkCreateDescriptorSetLayout(self->device, &layout_info, NULL,
            &self->descriptor_set_layout) != VK_SUCCESS) {
        fatal("Failed to create descriptor set layout.");
    }

    VkSemaphoreCreateInfo semaphore_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    for (size_t i = 0; i < 2; i++) {
        vkCreateSemaphore(self->device, &semaphore_info, NULL,
                &self->image_available_semaphores[i]);
        vkCreateSemaphore(self->device, &semaphore_info, NULL,
                &self->draw_finished_semaphores[i]);
        vkCreateFence(self->device, &fence_info, NULL,
                &self->commands_executed_fences[i]);
    }

    render_swapchain_dependent_init(self, window);
    return self;
}

static void render_swapchain_dependent_init(Render* self, GLFWwindow* window)
{
    self->swapchain = create_swapchain(
            self->physical_device,
            self->device,
            self->surface,
            self->graphics_family,
            self->present_family,
            window,
            &self->swapchain_format,
            &self->swapchain_extent,
            &self->swapchain_images
    );
    self->swapchain_image_views = create_swapchain_image_views(
            self->device,
            self->swapchain_format,
            self->swapchain_images
    );
    self->render_pass = create_render_pass(
            self->swapchain_format,
            self->physical_device,
            self->device
    );
    self->graphics_pipeline = create_graphics_pipeline(
            self->device,
            self->swapchain_extent,
            SAMPLE_COUNT,
            self->descriptor_set_layout,
            self->render_pass,
            &self->graphics_pipeline_layout
    );
    self->color_image = create_color_image(
            self->device,
            self->physical_device,
            self->swapchain_format,
            self->swapchain_extent,
            SAMPLE_COUNT,
            self->graphics_command_pool,
            self->graphics_queue,
            &self->color_image_view,
            &self->color_image_memory
    );
    self->depth_image = create_depth_image(
            self->device,
            self->physical_device,
            self->swapchain_extent,
            SAMPLE_COUNT,
            self->graphics_command_pool,
            self->graphics_queue,
            &self->depth_image_view,
            &self->depth_image_memory
    );
    self->framebuffers = create_framebuffers(
            self->device,
            self->color_image_view,
            self->depth_image_view,
            self->swapchain_image_views,
            self->render_pass,
            self->swapchain_extent
    );

    VkDescriptorType descriptor_types[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
    self->descriptor_pool = create_descriptor_pool(
            self->device, descriptor_types, 1
    );
    
    self->uniform_buffers = create_uniform_buffers(
            self->physical_device,
            self->device,
            &self->uniform_buffers_memories
    );
    self->descriptor_sets = create_descriptor_sets(
            self->device,
            self->descriptor_pool,
            self->descriptor_set_layout,
            self->uniform_buffers
    );

    // Upload projection matrices
    Uniform uniform;
    // TODO query window size
    mat4 view;
    vec3 eye = {4.0, 1.0, -10.0};
    vec3 up = {0.0, 1.0, 0.0};
    vec3 center = {4.0, 1.0, 0.0};
    glm_lookat(eye, center, up, view);

    mat4 proj;
    glm_perspective_default(
            self->swapchain_extent.width / (float) self->swapchain_extent.height,
            proj);
    proj[0][0] *= -1;
    proj[1][1] *= -1;

    glm_mat4_mul(proj, view, uniform.view_proj);

    for (size_t i=0; i < 2; i++) {
        upload_to_device_local_buffer(
                (void*) &uniform,
                sizeof(uniform),
                self->uniform_buffers[i],
                self->physical_device,
                self->device,
                self->graphics_queue,
                self->graphics_command_pool
        );
    }

    self->current_frame = 0;
}

void render_upload_map_mesh(
        Render* self, Vertex* vertices, size_t vertex_count,
        uint16_t* indices, size_t index_count) {
    self->vertex_buffer = device_local_buffer_from_data(
            (void*) vertices,
            sizeof(Vertex) * vertex_count,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            self->physical_device,
            self->device,
            self->graphics_queue,
            self->graphics_command_pool,
            &self->vertex_buffer_memory
    );
    self->index_buffer = device_local_buffer_from_data(
            (void*) indices,
            sizeof(uint16_t) * index_count,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            self->physical_device,
            self->device,

            self->graphics_queue,
            self->graphics_command_pool,
            &self->index_buffer_memory
    );
}

static void cleanup_swapchain(Render* self)
{
    mem_free(self->descriptor_sets);
    vkDestroyDescriptorPool(self->device, self->descriptor_pool, NULL);

    for (size_t i=0; i < 2; i++) {
        vkDestroySemaphore(
                self->device, self->image_available_semaphores[i], NULL);
        vkDestroySemaphore(
                self->device, self->draw_finished_semaphores[i], NULL);
        vkDestroyFence(self->device, self->commands_executed_fences[i], NULL);
    }

    for (size_t i=0; i < 2; i++) {
        vkDestroyFramebuffer(self->device, self->framebuffers[i], NULL);
    }
    mem_free(self->framebuffers);
    
    vkDestroyImageView(self->device, self->depth_image_view, NULL);
    vkDestroyImage(self->device, self->depth_image, NULL);
    vkFreeMemory(self->device, self->depth_image_memory, NULL);

    vkDestroyImageView(self->device, self->color_image_view, NULL);
    vkDestroyImage(self->device, self->color_image, NULL);
    vkFreeMemory(self->device, self->color_image_memory, NULL);

    vkDestroyPipeline(self->device, self->graphics_pipeline, NULL);
    vkDestroyPipelineLayout(self->device, self->graphics_pipeline_layout, NULL);

    vkDestroyRenderPass(self->device, self->render_pass, NULL);

    for (uint32_t i=0; i < 2; i++) {
        vkDestroyImageView(self->device, self->swapchain_image_views[i], NULL);
    }

    mem_free(self->swapchain_image_views);
    mem_free(self->swapchain_images);

    for (size_t i=0; i < 2; i++) {
        vkDestroyBuffer(self->device, self->uniform_buffers[i], NULL);
        vkFreeMemory(self->device, self->uniform_buffers_memories[i], NULL);
    }
    mem_free(self->uniform_buffers);
    mem_free(self->uniform_buffers_memories);

    vkDestroySwapchainKHR(self->device, self->swapchain, NULL);
}

void render_destroy(Render* self)
{
    vkDeviceWaitIdle(self->device);
    cleanup_swapchain(self);

    mem_free(self->command_buffers);

    vkDestroyDescriptorSetLayout(self->device, self->descriptor_set_layout, NULL);

    if (self->index_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(self->device, self->index_buffer, NULL);
        vkFreeMemory(self->device, self->index_buffer_memory, NULL);
    }

    if (self->vertex_buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(self->device, self->vertex_buffer, NULL);
        vkFreeMemory(self->device, self->vertex_buffer_memory, NULL);
    }

    vkDestroyCommandPool(self->device, self->graphics_command_pool, NULL);
    vkDestroyDevice(self->device, NULL);
    vkDestroySurfaceKHR(self->instance, self->surface, NULL);
    vkDestroyInstance(self->instance, NULL);

    glfwDestroyWindow(self->window);
    glfwTerminate();

    mem_free(self);
}

void render_draw_frame(Render* self, GLFWwindow* window) {
    PushConstants push_constants = {
        .model = GLM_MAT4_IDENTITY_INIT,
    };
    // TODO change to one-frame command buffer
    self->command_buffers = record_command_buffers(
            self->device,
            self->graphics_command_pool,
            self->render_pass,
            self->framebuffers,
            self->swapchain_extent,
            self->graphics_pipeline_layout,
            self->graphics_pipeline,
            self->vertex_buffer,
            self->index_buffer,
            3,
            self->descriptor_sets,
            &push_constants
    );

    size_t current_frame = self->current_frame;
    vkWaitForFences(
            self->device, 1, &self->commands_executed_fences[current_frame],
            VK_TRUE, UINT64_MAX);
    uint32_t image_index;
    VkResult acquire_image_result =
        vkAcquireNextImageKHR(self->device, self->swapchain, UINT64_MAX,
                self->image_available_semaphores[current_frame], VK_NULL_HANDLE,
                &image_index);

    if (acquire_image_result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_swapchain(self, window);
        return;
    } else if (acquire_image_result != VK_SUCCESS &&
            acquire_image_result != VK_SUBOPTIMAL_KHR) {
        fatal("Failed to acquire swapchain image.");
    }

    VkPipelineStageFlags wait_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &self->image_available_semaphores[current_frame],
        .pWaitDstStageMask = &wait_mask,
        .commandBufferCount = 1,
        .pCommandBuffers = &self->command_buffers[image_index],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &self->draw_finished_semaphores[current_frame],
    };

    vkResetFences(self->device, 1, &self->commands_executed_fences[current_frame]);

    if (vkQueueSubmit(self->graphics_queue, 1, &submit_info,
            self->commands_executed_fences[current_frame]) != VK_SUCCESS) {
        fatal("Failed to submit draw command buffer.");
    }

    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &self->draw_finished_semaphores[current_frame],
        .swapchainCount = 1,
        .pSwapchains = &self->swapchain,
        .pImageIndices = &image_index,
    };

    VkResult present_result = vkQueuePresentKHR(self->present_queue, &present_info);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
            present_result == VK_SUBOPTIMAL_KHR) {
        recreate_swapchain(self, window);
    } else if (present_result != VK_SUCCESS) {
        fatal("Failed to present swapchain image.");
    }

    self->current_frame = (current_frame + 1) % 2;
}

static VkInstance create_instance()
{
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = APP_NAME,
        .applicationVersion = VK_MAKE_VERSION(0, 0, 1),
        .pEngineName = ENGINE_NAME,
        .engineVersion = VK_MAKE_VERSION(0, 0, 1),
        .apiVersion = VK_API_VERSION_1_0,
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

    VkInstance instance;
    if (vkCreateInstance(&create_info, NULL, &instance) != VK_SUCCESS) {
        fatal("Failed to create instance");
    }
    return instance;
}

static VkPhysicalDevice pick_physical_device(
        const VkInstance instance, const VkSurfaceKHR surface,
        uint32_t *const o_graphics_family, uint32_t *const o_present_family)
{
    uint32_t dev_count;
    if (vkEnumeratePhysicalDevices(instance, &dev_count, NULL) !=
            VK_SUCCESS) fatal("Failed to enumerate physical devices.");
    VkPhysicalDevice *devices = malloc_nofail(
                                        sizeof(VkPhysicalDevice) * dev_count);
    if (vkEnumeratePhysicalDevices(instance, &dev_count, devices) !=
            VK_SUCCESS) fatal("Failed to enumerate physical devices.");

    VkPhysicalDevice result = VK_NULL_HANDLE;
    int graphics;
    int present;
    for (size_t i=0; i < dev_count; i++) {
        VkPhysicalDevice device = devices[i];

        // Find graphics and present queue families
        // TODO use a specialized transfer queue
        uint32_t queue_family_count;
        vkGetPhysicalDeviceQueueFamilyProperties(
                device, &queue_family_count, NULL);
        VkQueueFamilyProperties *queue_families =
           malloc_nofail(sizeof(VkQueueFamilyProperties) * queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
                                 device, &queue_family_count, queue_families);

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
                                device, j, surface, &present_support);

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
        vkEnumerateDeviceExtensionProperties(device, NULL, &ext_count, NULL);
        VkExtensionProperties *available_extensions =
                       malloc_nofail(sizeof(VkExtensionProperties) * ext_count);
        vkEnumerateDeviceExtensionProperties(
                               device, NULL, &ext_count, available_extensions);

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
                                device, surface, &format_count, NULL);
        if (format_count == 0) continue;

        // Make sure that present mode count isn't 0
        // TODO check for actual needed present mode
        uint32_t present_mode_count;
        vkGetPhysicalDeviceSurfacePresentModesKHR(
                            device, surface, &present_mode_count, NULL);
        if (present_mode_count == 0) continue;

        // Multisampling
        VkPhysicalDeviceFeatures supported_features;
        vkGetPhysicalDeviceFeatures(device, &supported_features);
        if (!supported_features.samplerAnisotropy) continue;

        result = device;
        break;
    }

    mem_free(devices); 
    if (result == VK_NULL_HANDLE) {
        fatal("Failed to find a suitable physical device.");
    } else {
        *o_graphics_family = (uint32_t) graphics;
        *o_present_family = (uint32_t) present;
        return result;
    }
}

static VkDevice create_logical_device(
        VkPhysicalDevice physical_device, VkSurfaceKHR surface,
        uint32_t graphics_family, uint32_t present_family,
        VkQueue* o_graphics_queue, VkQueue* o_present_queue
) {
    uint32_t queue_count;
    if (graphics_family != present_family) {
        queue_count = 2;
    } else {
        queue_count = 1;
    }

    VkDeviceQueueCreateInfo* queue_create_infos =
        malloc_nofail(sizeof(*queue_create_infos) * queue_count);

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo graphics_queue_create_info =  {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = graphics_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    queue_create_infos[0] = graphics_queue_create_info;
    
    if (queue_count > 1) {
        VkDeviceQueueCreateInfo present_queue_create_info = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = present_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };
        queue_create_infos[1] = present_queue_create_info;
    }

    VkPhysicalDeviceFeatures features = {
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

    VkDevice device;
    if (vkCreateDevice(
            physical_device, &device_create_info, NULL, &device) !=
            VK_SUCCESS) {
        fatal("Failed to create logical device.");
    }

    vkGetDeviceQueue(
        device, graphics_family, 0, o_graphics_queue);
    vkGetDeviceQueue(
        device, present_family, 0, o_present_queue);

    mem_free(queue_create_infos);

    return device;
}

static VkSwapchainKHR create_swapchain(const VkPhysicalDevice physical_device,
        const VkDevice device,
        const VkSurfaceKHR surface, uint32_t graphics_family,
        uint32_t present_family, GLFWwindow* const window,
        VkFormat* const o_swapchain_format,
        VkExtent2D* const o_swapchain_extent,
        VkImage* *const o_images) {
    // Choose swap surface format
    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(
                                physical_device, surface, &format_count, NULL);
    VkSurfaceFormatKHR *const formats = 
            malloc_nofail(sizeof(VkSurfaceFormatKHR) * format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(
                physical_device, surface, &format_count, formats);

    // NOTE assumes that format_count > 0
    VkSurfaceFormatKHR format;
    if (format_count == 1 && formats->format == VK_FORMAT_UNDEFINED) {
        VkSurfaceFormatKHR deflt = {
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        };
        format = deflt;
        goto format_chosen;
    }

    for (int i=0; i < format_count; i++) {
        if (formats[i].format == VK_FORMAT_B8G8R8A8_SNORM &&
                formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            format = formats[i];
            goto format_chosen;
        }
    }

    // TODO check if we can really use any format
    format = *formats;
format_chosen:
    mem_free(formats);

    // Choose present mode
    uint32_t present_mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(
                            physical_device, surface, &present_mode_count, NULL);
    VkPresentModeKHR* present_modes =
        malloc_nofail(sizeof(VkPresentModeKHR) * present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device, surface, &present_mode_count, present_modes);

    // NOTE assumes that present_mode_count > 0
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
                            physical_device, surface, &capabilities);
    VkExtent2D extent;
    if (capabilities.currentExtent.width != UINT32_MAX) {
        extent = capabilities.currentExtent;
    } else {
        int width;
        int height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D actual_extent = {
            (uint32_t) width,
            (uint32_t) height,
        };
        // TODO clamp function?
        actual_extent.width = MAX(capabilities.minImageExtent.width,
                                 MIN(capabilities.maxImageExtent.width,
                                     actual_extent.width));

        actual_extent.height = MAX(capabilities.minImageExtent.height,
                                  MIN(capabilities.maxImageExtent.height,
                                      actual_extent.height));
        extent = actual_extent;
    }

    struct VkSwapchainCreateInfoKHR create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = 2, // ! hard condition
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
    *o_swapchain_format = format.format;
    *o_swapchain_extent = extent;

    if (graphics_family != present_family) {
        uint32_t queue_family_indices[] = {
            graphics_family,
            present_family,
        };

        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;
        create_info.pQueueFamilyIndices = queue_family_indices;
    } else {
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VkSwapchainKHR swapchain;
    if (vkCreateSwapchainKHR(
                device, &create_info, NULL, &swapchain) != VK_SUCCESS) {
        fatal("Failed to create swapchain.");
    }

    uint32_t image_count;
    vkGetSwapchainImagesKHR(device, swapchain, &image_count, NULL);
    *o_images = malloc_nofail(sizeof(VkImage) * (image_count));
    DBASSERT(image_count == 2);
    vkGetSwapchainImagesKHR(device, swapchain, &image_count, *o_images);

    return swapchain;
}

// TODO merge with create_swapchain
static VkImageView* create_swapchain_image_views(VkDevice device, VkFormat format,
        VkImage* images) {
    VkImageView* views = malloc_nofail(sizeof(VkImageView) * 2);

    for (uint32_t i=0; i < 2; i++) {
        VkImageViewCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components.r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .components.g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .components.b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .components.a = VK_COMPONENT_SWIZZLE_IDENTITY,
            .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .subresourceRange.baseMipLevel = 0,
            .subresourceRange.levelCount = 1,
            .subresourceRange.baseArrayLayer = 0,
            .subresourceRange.layerCount = 1,
        };
        if (vkCreateImageView(
              device, &create_info, NULL, &views[i]) != VK_SUCCESS) {
            fatal("Failed to create image views.");
        }
    }

    return views;
}

static VkFormat find_depth_format(VkPhysicalDevice physical_device) {
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
                physical_device, candidates[i], &properties);
        if (properties.optimalTilingFeatures &
                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            depth_format = candidates[i];
            found = true;
            break;
        }
    }

    if (!found) fatal("Failed to find format that supports depth buffering.");

    return depth_format;
}

static VkRenderPass create_render_pass(const VkFormat swapchain_format,
        const VkPhysicalDevice physical_device, const VkDevice device)
{
    struct VkAttachmentDescription color_attachment = {
        .format = swapchain_format,
        .samples = SAMPLE_COUNT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .flags = 0,
    };

    VkFormat depth_format = find_depth_format(physical_device);

    struct VkAttachmentDescription depth_attachment = {
        .format = depth_format,
        .samples = SAMPLE_COUNT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    struct VkAttachmentDescription color_attachment_resolve = {
        .format = swapchain_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    enum {attachment_count = 3};
    struct VkAttachmentDescription attachments[attachment_count] = {
        color_attachment, depth_attachment, color_attachment_resolve,
    };

    struct VkAttachmentReference color_attachment_ref = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    struct VkAttachmentReference depth_attachment_ref = {
        .attachment = 1,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    struct VkAttachmentReference color_attachment_resolve_ref = {
        .attachment = 2,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
        .pDepthStencilAttachment = &depth_attachment_ref,
        .pResolveAttachments = &color_attachment_resolve_ref,
    };

    struct VkSubpassDependency dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    VkRenderPassCreateInfo render_pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = attachment_count,
        .pAttachments = attachments,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    VkRenderPass render_pass;
    if (vkCreateRenderPass(
           device, &render_pass_info, NULL, &render_pass) !=
           VK_SUCCESS) {
        fatal("Failed to create render pass.");
    }
    return render_pass;
}

static VkShaderModule create_shader_module(
                    VkDevice device, const char *const code, const size_t size)
{
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = (const uint32_t*) code,
    };

    VkShaderModule shader_module;
    if (vkCreateShaderModule(
            device, &create_info, NULL, &shader_module) !=
            VK_SUCCESS) {
        fatal("Failed to create shader module.");
    }
    return shader_module;
}

static VkPipeline create_graphics_pipeline(
        VkDevice device,
        VkExtent2D swapchain_extent,
        VkSampleCountFlagBits sample_count,
        VkDescriptorSetLayout descriptor_set_layout,
        VkRenderPass render_pass,
        VkPipelineLayout* o_layout)
{ 
    char *vertex_shader_code; 
    size_t vertex_shader_code_size; 
    if (read_binary_file(
            VERTEX_SHADER_PATH, &vertex_shader_code, &vertex_shader_code_size)) {
        fatal("Failed to read vertex shader");
    }

    char *fragment_shader_code;
    size_t fragment_shader_code_size;
    if (read_binary_file(
            FRAGMENT_SHADER_PATH, &fragment_shader_code,
            &fragment_shader_code_size)) {
        fatal("Failed to read fragment shader");
    }

    VkShaderModule vertex_shader_module = create_shader_module(
                        device, vertex_shader_code, vertex_shader_code_size);
    VkShaderModule fragment_shader_module = create_shader_module(
                    device, fragment_shader_code, fragment_shader_code_size);

    struct VkPipelineShaderStageCreateInfo vertex_shader_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertex_shader_module,
        .pName = "main",
    };

    struct VkPipelineShaderStageCreateInfo fragment_shader_stage_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragment_shader_module,
        .pName = "main",
    };

    enum {shader_stage_count = 2};
    struct VkPipelineShaderStageCreateInfo shader_stages[shader_stage_count] = {
        vertex_shader_stage_info, fragment_shader_stage_info,
    };

    VkVertexInputBindingDescription vertex_input_binding_description = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    enum { v_attr_desc_count = 2 };
    VkVertexInputAttributeDescription attribute_descriptions[v_attr_desc_count];

    attribute_descriptions[0].binding = 0;
    attribute_descriptions[0].location = 0;
    attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT; 
    attribute_descriptions[0].offset = offsetof(Vertex, position);

    attribute_descriptions[1].binding = 0;
    attribute_descriptions[1].location = 1;
    attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribute_descriptions[1].offset = offsetof(Vertex, color);

    struct VkPipelineVertexInputStateCreateInfo vertex_input_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_input_binding_description,
        .vertexAttributeDescriptionCount = v_attr_desc_count,
        .pVertexAttributeDescriptions = attribute_descriptions,
    };

    struct VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    const VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width  = (float) swapchain_extent.width,
        .height = (float) swapchain_extent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const VkRect2D scissor = {
        .offset = {0, 0},
        .extent = swapchain_extent,
    };

    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    const VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .lineWidth = 1.0f,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_FRONT_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
    };

    const VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = VK_FALSE,
        .rasterizationSamples = sample_count,
    };

    const VkPipelineColorBlendAttachmentState color_blend_attachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_A_BIT |
                          VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_R_BIT,
        .blendEnable = VK_FALSE,
    };

    const VkPipelineColorBlendStateCreateInfo color_blending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
    };

    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };

    const VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
    };

    VkPipelineLayout pipeline_layout;
    if (vkCreatePipelineLayout(
                            device,
                            &pipeline_layout_info,
                            NULL,
                            &pipeline_layout) != VK_SUCCESS) {
        fatal("Failed to create pipeline layout.");
    }

    const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
    };

    const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shader_stages,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &color_blending,
        .pDepthStencilState = &depth_stencil,
        .layout = pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
    };

    VkPipeline graphics_pipeline;
    if (vkCreateGraphicsPipelines(
                              device,
                              VK_NULL_HANDLE,
                              1,
                              &pipeline_info,
                              NULL,
                              &graphics_pipeline) != VK_SUCCESS) {
        fatal("Failed to create graphics pipeline.");
    }

    vkDestroyShaderModule(device, fragment_shader_module, NULL);
    vkDestroyShaderModule(device, vertex_shader_module, NULL);
    mem_free(vertex_shader_code);
    mem_free(fragment_shader_code);

    *o_layout = pipeline_layout;
    return graphics_pipeline;
}

static VkCommandBuffer begin_one_time_command_buffer(
        VkDevice device, VkCommandPool command_pool)
{
    VkCommandBufferAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandPool = command_pool,
        .commandBufferCount = 1,
    };

    VkCommandBuffer command_buffer;
    vkAllocateCommandBuffers(device, &allocate_info, &command_buffer);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkBeginCommandBuffer(command_buffer, &begin_info);

    return command_buffer;
}

static void finish_one_time_command_buffer(VkDevice device, VkQueue queue,
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

    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);
}

static VkImage create_color_image(
        VkDevice device, VkPhysicalDevice physical_device, VkFormat format,
        VkExtent2D extent, VkSampleCountFlagBits sample_count,
        VkCommandPool command_pool, VkQueue queue, VkImageView* o_view,
        VkDeviceMemory* o_memory
) {
    VkImageCreateInfo image_create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .extent.width = extent.width,
        .extent.height = extent.height,
        .extent.depth = 1,
        .mipLevels = 1,
        .arrayLayers = 1,
        .format = format,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .samples = sample_count,
    };

    VkImage image;
    if (vkCreateImage(device, &image_create_info, NULL, &image) != VK_SUCCESS) {
        fatal("Failed to create image.");
    }

    VkMemoryRequirements memory_requirements;
    vkGetImageMemoryRequirements(device, image, &memory_requirements);

    int memory_type_index = find_memory_type(
            physical_device,
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

    VkDeviceMemory image_memory;
    if (vkAllocateMemory(device, &allocate_info, NULL, &image_memory)
            != VK_SUCCESS) {
        fatal("Failed to allocate image memory.");
    }

    vkBindImageMemory(device, image, image_memory, 0);
    *o_memory = image_memory;

    VkImageViewCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components.r = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.g = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.b = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.a = VK_COMPONENT_SWIZZLE_IDENTITY,
        .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .subresourceRange.baseMipLevel = 0,
        .subresourceRange.levelCount = 1,
        .subresourceRange.baseArrayLayer = 0,
        .subresourceRange.layerCount = 1,
    };

    VkImageView image_view;
    if (vkCreateImageView(device, &create_info, NULL,
            &image_view) != VK_SUCCESS) {
        fatal("Failed to create image views.");
    }
    *o_view = image_view;

    VkCommandBuffer command_buffer = begin_one_time_command_buffer(
            device, command_pool);

    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange.baseMipLevel = 0,
        .subresourceRange.levelCount = 1,
        .subresourceRange.baseArrayLayer = 0,
        .subresourceRange.layerCount = 1,
        .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    vkCmdPipelineBarrier(command_buffer, 
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 
        0, 
        0, NULL,
        0, NULL,
        1, &barrier);

    finish_one_time_command_buffer(device, queue, command_buffer, command_pool);

    return image;
}

static VkImage create_depth_image(
        VkDevice device, VkPhysicalDevice physical_device,
        VkExtent2D extent, VkSampleCountFlagBits sample_count,
        VkCommandPool command_pool, VkQueue queue, VkImageView* o_view,
        VkDeviceMemory* o_memory
) {
    VkFormat format = find_depth_format(physical_device);

    VkImageCreateInfo image_create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .extent.width = extent.width,
        .extent.height = extent.height,
        .extent.depth = 1,
        .mipLevels = 1,
        .arrayLayers = 1,
        .format = format,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .samples = sample_count,
    };

    VkImage image;
    if (vkCreateImage(device, &image_create_info, NULL, &image) != VK_SUCCESS) {
        fatal("Failed to create image.");
    }

    VkMemoryRequirements memory_requirements;
    vkGetImageMemoryRequirements(device, image, &memory_requirements);

    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    int memory_type_index = -1;
    // TODO move into separate function?
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if ((memory_requirements.memoryTypeBits & (1 << i)) &&
            (memory_properties.memoryTypes[i].propertyFlags &
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            memory_type_index = (int) i;
            break;
        }
    }
    if (memory_type_index < 0) {
        fatal("Failed to find suitable memory type for image creation.");
    }

    VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex = (uint32_t) memory_type_index,
    };

    VkDeviceMemory image_memory;
    if (vkAllocateMemory(device, &allocate_info, NULL, &image_memory)
            != VK_SUCCESS) {
        fatal("Failed to allocate image memory.");
    }

    vkBindImageMemory(device, image, image_memory, 0);
    *o_memory = image_memory;

    VkImageViewCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components.r = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.g = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.b = VK_COMPONENT_SWIZZLE_IDENTITY,
        .components.a = VK_COMPONENT_SWIZZLE_IDENTITY,
        .subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
        .subresourceRange.baseMipLevel = 0,
        .subresourceRange.levelCount = 1,
        .subresourceRange.baseArrayLayer = 0,
        .subresourceRange.layerCount = 1,
    };

    VkImageView image_view;
    if (vkCreateImageView(device, &create_info, NULL,
            &image_view) != VK_SUCCESS) {
        fatal("Failed to create image views.");
    }
    *o_view = image_view;

    VkCommandBuffer command_buffer = begin_one_time_command_buffer(
            device, command_pool);

    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange.baseMipLevel = 0,
        .subresourceRange.levelCount = 1,
        .subresourceRange.baseArrayLayer = 0,
        .subresourceRange.layerCount = 1,
        .subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    };

    vkCmdPipelineBarrier(command_buffer, 
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 
        0, 
        0, NULL,
        0, NULL,
        1, &barrier);

    finish_one_time_command_buffer(device, queue, command_buffer, command_pool);

    return image;
}

static VkFramebuffer* create_framebuffers(
        VkDevice device,
        VkImageView color_image_view, VkImageView depth_image_view,
        VkImageView* swapchain_image_views, VkRenderPass render_pass,
        VkExtent2D swapchain_extent)
{
    VkFramebuffer* framebuffers = malloc_nofail(sizeof(VkFramebuffer) * 2);
    for (size_t i = 0; i < 2; ++i) {
        enum { attachment_count = 3 };
        VkImageView attachments[attachment_count] = {
            color_image_view,
            depth_image_view,
            swapchain_image_views[i]
        };

        VkFramebufferCreateInfo framebuffer_info = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = attachment_count,
            .pAttachments = attachments,
            .width = swapchain_extent.width,
            .height = swapchain_extent.height,
            .layers = 1,
        };

        if (vkCreateFramebuffer(device, &framebuffer_info, NULL,
                &framebuffers[i]) != VK_SUCCESS) {
            fatal("Failed to create framebuffer.");
        }
    }
    return framebuffers;
}

static VkDescriptorPool create_descriptor_pool(
        VkDevice device, VkDescriptorType* descriptor_types,
        size_t descriptor_type_count
) {
    VkDescriptorPoolSize* pool_sizes = malloc_nofail(
            sizeof(VkDescriptorPoolSize) * descriptor_type_count);

    for (size_t i=0; i < descriptor_type_count; i++) {
        pool_sizes[i].type = descriptor_types[i];
        pool_sizes[i].descriptorCount = 2;
    }

    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = descriptor_type_count,
        .pPoolSizes = pool_sizes,
        .maxSets = 2,
    };

    VkDescriptorPool descriptor_pool;
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &descriptor_pool)
            != VK_SUCCESS) {
        fatal("Failed to create descriptor pool.");
    }

    mem_free(pool_sizes);

    return descriptor_pool;
}

static int find_memory_type(
        VkPhysicalDevice physical_device,
        VkMemoryRequirements memory_requirements,
        VkMemoryPropertyFlags required_properties
) { 
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

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

static int create_buffer(
        VkPhysicalDevice physical_device,
        VkDevice device,
        size_t size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, VkBuffer *buffer,
        VkDeviceMemory *bufferMemory)
{
    VkDeviceSize device_size = size;
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = device_size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    if (vkCreateBuffer(device, &buffer_info, NULL, buffer) != VK_SUCCESS) {
        return 1;
    }

    VkMemoryRequirements memory_requirements;
    vkGetBufferMemoryRequirements(device, *buffer, &memory_requirements);

    VkMemoryAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memory_requirements.size,
        .memoryTypeIndex =
            find_memory_type(physical_device, memory_requirements, properties),
    };

    if (vkAllocateMemory(device, &allocate_info, NULL, bufferMemory)
            != VK_SUCCESS) {
        return 2;
    }

    vkBindBufferMemory(device, *buffer, *bufferMemory, 0);
    return 0;
}

static void copy_buffer(
        VkDevice device,
        VkQueue queue,
        VkCommandPool command_pool,
        VkBuffer src_buffer, VkBuffer dst_buffer,
        VkDeviceSize device_size
) {
    VkCommandBuffer command_buffer = begin_one_time_command_buffer(
        device, command_pool);

    VkBufferCopy copy_region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = device_size,
    };
    vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);
    finish_one_time_command_buffer(device, queue, command_buffer, command_pool);
}

static void upload_to_device_local_buffer(
        void* data,
        size_t size,
        VkBuffer destination,
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkQueue queue,
        VkCommandPool command_pool
) {
    VkDeviceSize device_size = size;

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;
    create_buffer(
            physical_device,
            device,
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &staging_buffer,
            &staging_buffer_memory
    );

    void *staging_buffer_mapped;
    vkMapMemory(
            device, staging_buffer_memory, 0, device_size, 0,
            &staging_buffer_mapped);
    memcpy(staging_buffer_mapped, data, size);
    vkUnmapMemory(device, staging_buffer_memory);

    copy_buffer(
            device,
            queue,
            command_pool,
            staging_buffer, destination, device_size
    );

    vkDestroyBuffer(device, staging_buffer, NULL);
    vkFreeMemory(device, staging_buffer_memory, NULL);
}

static VkBuffer device_local_buffer_from_data(
        void* data,
        size_t size,
        VkBufferUsageFlags usage,
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkQueue queue,
        VkCommandPool command_pool,
        VkDeviceMemory *o_memory
) {
    VkBuffer device_local_buffer;
    VkDeviceMemory device_local_buffer_memory;
    create_buffer(
            physical_device,
            device,
            size,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &device_local_buffer,
            &device_local_buffer_memory
    );
    upload_to_device_local_buffer(
            data,
            size,
            device_local_buffer,
            physical_device,
            device,
            queue,
            command_pool
    );

    *o_memory = device_local_buffer_memory;
    return device_local_buffer;
}

VkBuffer* create_uniform_buffers(
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkDeviceMemory* *const o_memories
) {
    VkBuffer* uniform_buffers = malloc_nofail(
                                sizeof(VkBuffer) * 2);
    VkDeviceMemory* memories = malloc_nofail(
            sizeof(VkDeviceMemory) * 2);
    for (uint32_t i=0; i < 2; i++) {
        create_buffer(
                physical_device,
                device,
                sizeof(Uniform),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                &uniform_buffers[i],
                &memories[i]
        );
    }

    *o_memories = memories;
    return uniform_buffers;
}

static VkDescriptorSet* create_descriptor_sets(
        VkDevice device,
        VkDescriptorPool descriptor_pool,
        VkDescriptorSetLayout descriptor_set_layout,
        VkBuffer* uniform_buffers
) {
    VkDescriptorSetLayout* layout_copies = malloc_nofail(
            sizeof(VkDescriptorSetLayout) * 2);
    for (size_t i=0; i < 2; i++) {
        layout_copies[i] = descriptor_set_layout;
    }

    VkDescriptorSetAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 2,
        .pSetLayouts = layout_copies,
    };

    VkDescriptorSet* descriptor_sets = malloc_nofail(
            sizeof(VkDescriptorSet) * 2);
    if (vkAllocateDescriptorSets(device, &allocate_info, descriptor_sets)
            != VK_SUCCESS) {
        fatal("Failed to allocate descriptor sets.");
    }

    mem_free(layout_copies);

    for (size_t i = 0; i < 2; i++) {
        VkDescriptorBufferInfo buffer_info = {
            .buffer = uniform_buffers[i],
            .offset = 0,
            .range = sizeof(Uniform),
        };

        // TODO replace with a loop that iterates over descriptor types?
        VkWriteDescriptorSet uniform_write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_sets[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &buffer_info,
        };

        vkUpdateDescriptorSets(
                device,
                1,
                &uniform_write,
                0,
                NULL);
    }

    return descriptor_sets;
}

static VkCommandBuffer* record_command_buffers(
        VkDevice device,
        VkCommandPool command_pool,
        VkRenderPass render_pass,
        VkFramebuffer* swapchain_framebuffers,
        VkExtent2D swapchain_extent,
        VkPipelineLayout graphics_pipeline_layout,
        VkPipeline graphics_pipeline,
        VkBuffer vertex_buffer,
        VkBuffer index_buffer,
        uint32_t index_count,
        VkDescriptorSet* descriptor_sets,
        PushConstants* push_constants
) {
    VkCommandBuffer* command_buffers = malloc_nofail(
            sizeof(VkCommandBuffer) * 2); 
    VkCommandBufferAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 2,
    };

    if (vkAllocateCommandBuffers(device, &allocate_info, command_buffers) !=
            VK_SUCCESS) fatal("Failed to allocate command buffers.");

    for (size_t i = 0; i < 2; i++) {
        VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
        };

        if (vkBeginCommandBuffer(command_buffers[i], &begin_info) !=
                VK_SUCCESS) {
            fatal("Failed to begin recording command buffer.");
        }

        VkClearValue clear_values[2] = {
            { .color = {0.0f, 0.0f, 0.0f, 1.0f} },
            { .depthStencil = {1.0f, 0} },
        };
        VkRenderPassBeginInfo render_pass_info = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render_pass,
            .framebuffer = swapchain_framebuffers[i],
            .renderArea.offset = {0, 0},
            .renderArea.extent = swapchain_extent,
            .clearValueCount = 2,
            .pClearValues = clear_values,
        };

        vkCmdBeginRenderPass(command_buffers[i], &render_pass_info,
                             VK_SUBPASS_CONTENTS_INLINE);

        vkCmdPushConstants(
                command_buffers[i],
                graphics_pipeline_layout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                sizeof(PushConstants),
                push_constants);

        vkCmdBindPipeline(command_buffers[i],
                VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command_buffers[i], 0, 1, &vertex_buffer,
                &offset);
        vkCmdBindIndexBuffer(command_buffers[i], index_buffer, 0,
                VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(command_buffers[i],
                VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_layout, 0, 1,
                &descriptor_sets[i], 0, NULL);
        vkCmdDrawIndexed(command_buffers[i], index_count, 1, 0, 0, 0);
        vkCmdEndRenderPass(command_buffers[i]);

        if (vkEndCommandBuffer(command_buffers[i]) != VK_SUCCESS) {
            fatal("Failed to record command buffer.");
        }
    }

    return command_buffers;
}

static void recreate_swapchain(Render* self, GLFWwindow* window)
{
    int width = 0, height = 0;
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(self->device);

    cleanup_swapchain(self);

    render_swapchain_dependent_init(self, window);
}

void render_test()
{
    Render* render = render_init();

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

    while (!glfwWindowShouldClose(render->window)) {
        glfwPollEvents();
        render_draw_frame(render, render->window);
    }

    render_destroy(render);
}

int main()
{
    mem_init(MBS(24));

    render_test();
    mem_check();
    
    mem_shutdown();

    printf("%s", "Success!\n");
    return EXIT_SUCCESS;
}
