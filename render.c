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
static Mesh* g_meshes;
static size_t g_meshes_count;

typedef struct Node Node;
typedef struct Node {
    Node* parent;
    Node** children;
    uint32_t children_count;
    mat4 transform;
    Mesh* mesh;
} Node;
static Node* g_nodes;
static size_t g_nodes_count;

#define MAX_TEXTURES 50
static size_t g_texture_count;
static VkImage* g_texture_images;
static VkDeviceMemory* g_texture_image_memories;
static VkImageView* g_texture_image_views;
static VkDescriptorSet* g_texture_descriptor_sets;


typedef struct Uniform {
    mat4 view_proj;
} Uniform;

typedef struct PushConstants {
    mat4 model;
} PushConstants;

static void create_2d_image(VkPhysicalDevice physical_device,
        VkDevice device, uint32_t width, uint32_t height,
        VkSampleCountFlagBits samples, VkFormat format, VkImageTiling tiling,
        VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
        VkImage* image, VkDeviceMemory* memory);
static void create_2d_image_view(VkDevice device, VkImage image, VkFormat format,
        VkImageAspectFlags aspect_flags, VkImageView* image_view);
static VkCommandBuffer begin_one_time_command_buffer(
        VkDevice device, VkCommandPool command_pool);
static void submit_one_time_command_buffer(VkDevice device, VkQueue queue,
        VkCommandBuffer command_buffer, VkCommandPool command_pool);
static VkFormat find_depth_format(VkPhysicalDevice physical_device);
static VkShaderModule create_shader_module(
                    VkDevice device, const char* path);
static int find_memory_type(
        VkPhysicalDevice physical_device,
        VkMemoryRequirements memory_requirements,
        VkMemoryPropertyFlags required_properties
);
static int create_buffer(
        VkPhysicalDevice physical_device,
        VkDevice device,
        size_t size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, VkBuffer *buffer,
        VkDeviceMemory *bufferMemory);
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
static void upload_to_device_local_buffer(
        void* data,
        size_t size,
        VkBuffer destination,
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkQueue queue,
        VkCommandPool command_pool
);
static VkBuffer upload_data_to_staging_buffer(
    VkPhysicalDevice physical_device, VkDevice device, void* data,
    size_t size, VkDeviceMemory* staging_buffer_memory);

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

void create_attachment(Attachment* att, VkPhysicalDevice physical_device,
        VkDevice device, uint32_t width, uint32_t height, VkFormat format,
        VkImageUsageFlags usage)
{
    VkImageAspectFlags aspect_mask = 0;
    if (usage == VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        aspect_mask |= VK_IMAGE_ASPECT_DEPTH_BIT;
    } else {
        aspect_mask |= VK_IMAGE_ASPECT_COLOR_BIT;
    };

    create_2d_image(physical_device, device, width, height, VK_SAMPLE_COUNT_1_BIT, format,
        VK_IMAGE_TILING_OPTIMAL, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &att->image, &att->memory);
    create_2d_image_view(device, att->image, format, aspect_mask, &att->view);
}

typedef struct Render {
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
    VkImage swapchain_images[FRAMES_IN_FLIGHT];
    VkImageView swapchain_image_views[FRAMES_IN_FLIGHT];
    VkRenderPass render_pass;
    VkRenderPass offscreen_render_pass;
    VkDescriptorSetLayout view_proj_set_layout;
    VkDescriptorSetLayout texture_set_layout;
    VkPipelineLayout graphics_pipeline_layout;
    VkPipeline graphics_pipeline;
    VkPipeline offscreen_graphics_pipeline;
    VkCommandPool graphics_command_pool;

    Attachment color_att;
    Attachment depth_att;

    Attachment offscreen_position;
    Attachment offscreen_normal;
    Attachment offscreen_albedo;
    Attachment offscreen_depth;

    VkFramebuffer framebuffers[FRAMES_IN_FLIGHT];
    VkFramebuffer offscreen_framebuffer;

    VkDescriptorPool descriptor_pool;
    VkDescriptorPool texture_descriptor_pool;

    VkBuffer vertex_buffer;
    VkDeviceMemory vertex_buffer_memory;

    VkBuffer index_buffer;
    VkDeviceMemory index_buffer_memory;

    VkBuffer view_proj_buffers[FRAMES_IN_FLIGHT];
    VkDeviceMemory view_proj_buffers_memories[FRAMES_IN_FLIGHT];

    VkSampler texture_sampler;

    VkDescriptorSet view_proj_sets[FRAMES_IN_FLIGHT];
    VkDescriptorSet texture_sets[FRAMES_IN_FLIGHT];

    VkCommandBuffer command_buffers[FRAMES_IN_FLIGHT];

    VkFence commands_executed_fences[FRAMES_IN_FLIGHT];
    VkSemaphore image_available_semaphores[FRAMES_IN_FLIGHT];
    VkSemaphore draw_finished_semaphores[FRAMES_IN_FLIGHT];

    size_t current_frame;
} Render;

static void render_swapchain_dependent_init(Render* self);
static void recreate_swapchain(Render* self);

Render* render_init() {
    // CREATE WINDOW
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
     
    Render* self = (Render*) malloc_nofail(sizeof(Render));
    g_window = glfwCreateWindow(
            mode->width, mode->height, "Demo", monitor, NULL);
    glfwSetInputMode(g_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(g_window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

    // CREATE INSTANCE
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
    if (vkCreateInstance(&create_info, NULL, &self->instance) != VK_SUCCESS) {
        fatal("Failed to create instance");
    }

    // CREATE SURFACE
    if (glfwCreateWindowSurface(
            self->instance, g_window, NULL, &self->surface) != VK_SUCCESS) {
        fatal("Failed to create surface.");
    }

    // PICK PHYSICAL DEVICE
    uint32_t dev_count;
    if (vkEnumeratePhysicalDevices(self->instance, &dev_count, NULL) !=
            VK_SUCCESS) fatal("Failed to enumerate physical devices.");
    VkPhysicalDevice *devices = malloc_nofail(
                                        sizeof(VkPhysicalDevice) * dev_count);
    if (vkEnumeratePhysicalDevices(self->instance, &dev_count, devices) !=
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
                                device, j, self->surface, &present_support);

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
                                device, self->surface, &format_count, NULL);
        if (format_count == 0) continue;

        // Make sure that present mode count isn't 0
        // TODO check for actual needed present mode
        uint32_t present_mode_count;
        vkGetPhysicalDeviceSurfacePresentModesKHR(
                            device, self->surface, &present_mode_count, NULL);
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
        self->graphics_family = (uint32_t) graphics;
        self->present_family = (uint32_t) present;
        self->physical_device = result;
    }

    // CREATE LOGICAL DEVICE
    uint32_t queue_count;
    if (self->graphics_family != self->present_family) {
        queue_count = 2;
    } else {
        queue_count = 1;
    }

    VkDeviceQueueCreateInfo* queue_create_infos =
        malloc_nofail(sizeof(*queue_create_infos) * queue_count);

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo graphics_queue_create_info =  {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = self->graphics_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    queue_create_infos[0] = graphics_queue_create_info;
    
    // TODO use a separate queue for presentation (IMPORTANT!)
    if (queue_count > 1) {
        VkDeviceQueueCreateInfo present_queue_create_info = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = self->present_family,
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
            self->physical_device, &device_create_info, NULL, &self->device) !=
            VK_SUCCESS) fatal("Failed to create logical device.");

    vkGetDeviceQueue(
        self->device, self->graphics_family, 0, &self->graphics_queue);
    vkGetDeviceQueue(
        self->device, self->present_family, 0, &self->present_queue);
    mem_free(queue_create_infos);

    // CREATE COMMAND POOL
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = self->graphics_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    if (vkCreateCommandPool(self->device, &pool_info, NULL, 
            &self->graphics_command_pool) != VK_SUCCESS) {
        fatal("Failed to create command pool.");
    }

    self->vertex_buffer = VK_NULL_HANDLE;
    self->index_buffer = VK_NULL_HANDLE;

    // CREATE TEXTURE SAMPLER
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

    if (vkCreateSampler(self->device, &texture_sampler_info, NULL,
                &self->texture_sampler) != VK_SUCCESS) {
        fatal("Failed to create texture sampler.");
    }

    // CREATE DESCRIPTOR SET LAYOUTS AND PIPELINE LAYOUT
    VkDescriptorSetLayoutBinding view_proj_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    };
    VkDescriptorSetLayoutCreateInfo view_proj_set_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &view_proj_binding,
    };
    if (vkCreateDescriptorSetLayout(self->device, &view_proj_set_info, NULL,
            &self->view_proj_set_layout) != VK_SUCCESS) {
        fatal("Failed to create view-projection descriptor set layout.");
    }

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
    if (vkCreateDescriptorSetLayout(self->device, &texture_set_info, NULL,
            &self->texture_set_layout) != VK_SUCCESS) {
        fatal("Failed to create texture descriptor set layout.");
    }

    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };

    VkDescriptorSetLayout set_layouts[2] = {
        self->view_proj_set_layout, self->texture_set_layout,
    };

    const VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 2,
        .pSetLayouts = set_layouts,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
    };
    VkPipelineLayout pipeline_layout;
    if (vkCreatePipelineLayout(
                            self->device,
                            &pipeline_layout_info,
                            NULL,
                            &self->graphics_pipeline_layout) != VK_SUCCESS) {
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
    for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
        vkCreateSemaphore(self->device, &semaphore_info, NULL,
                &self->image_available_semaphores[i]);
        vkCreateSemaphore(self->device, &semaphore_info, NULL,
                &self->draw_finished_semaphores[i]);
        vkCreateFence(self->device, &fence_info, NULL,
                &self->commands_executed_fences[i]);
    }

    render_swapchain_dependent_init(self);
    return self;
}

static void render_swapchain_dependent_init(Render* self)
{
    // CREATE SWAPCHAIN
    // Choose swap surface format
    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(
                    self->physical_device, self->surface, &format_count, NULL);
    VkSurfaceFormatKHR *const formats = 
            malloc_nofail(sizeof(VkSurfaceFormatKHR) * format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(
                self->physical_device, self->surface, &format_count, formats);

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
            self->physical_device, self->surface, &present_mode_count, NULL);
    VkPresentModeKHR* present_modes =
        malloc_nofail(sizeof(VkPresentModeKHR) * present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        self->physical_device, self->surface, &present_mode_count, present_modes);

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
                            self->physical_device, self->surface, &capabilities);
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
        .surface = self->surface,
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
    self->swapchain_format = format.format;
    self->swapchain_extent = extent;

    if (self->graphics_family != self->present_family) {
        uint32_t queue_family_indices[] = {
            self->graphics_family,
            self->present_family,
        };

        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchain_create_info.queueFamilyIndexCount = 2;
        swapchain_create_info.pQueueFamilyIndices = queue_family_indices;
    } else {
        swapchain_create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    if (vkCreateSwapchainKHR(
            self->device, &swapchain_create_info, NULL, &self->swapchain) != VK_SUCCESS) {
        fatal("Failed to create swapchain.");
    }

    uint32_t image_count;
    vkGetSwapchainImagesKHR(self->device, self->swapchain, &image_count, NULL);
    DBASSERT(image_count == FRAMES_IN_FLIGHT);
    vkGetSwapchainImagesKHR(
          self->device, self->swapchain, &image_count, self->swapchain_images);

    for (uint32_t i=0; i < FRAMES_IN_FLIGHT; i++) {
        create_2d_image_view(self->device, self->swapchain_images[i],
            self->swapchain_format, VK_IMAGE_ASPECT_COLOR_BIT,
            &self->swapchain_image_views[i]);
    }

    // FINAL COMPOSITION RENDER PASS AND FRAMEBUFFER
    struct VkAttachmentDescription color_attachment = {
        .format = self->swapchain_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    VkFormat depth_format = find_depth_format(self->physical_device);
    struct VkAttachmentDescription depth_attachment = {
        .format = depth_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    enum {attachment_count = 2};
    struct VkAttachmentDescription attachments[attachment_count] = {
        color_attachment, depth_attachment,
    };
    struct VkAttachmentReference color_attachment_ref = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    struct VkAttachmentReference depth_attachment_ref = {
        .attachment = 1,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
        .pDepthStencilAttachment = &depth_attachment_ref,
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
           self->device, &render_pass_info, NULL, &self->render_pass) !=
           VK_SUCCESS) fatal("Failed to create render pass.");

    create_2d_image(self->physical_device, self->device, self->swapchain_extent.width,
            self->swapchain_extent.height, VK_SAMPLE_COUNT_1_BIT, depth_format,
            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &self->depth_att.image,
            &self->depth_att.memory);
    create_2d_image_view(self->device, self->depth_att.image, depth_format,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            &self->depth_att.view);

    for (int i=0; i < FRAMES_IN_FLIGHT; i++) {
        VkImageView attachments[2] = {
            self->swapchain_image_views[i],
            self->depth_att.view,
        };
        VkFramebufferCreateInfo framebuffer_info = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = self->render_pass,
            .attachmentCount = attachment_count,
            .pAttachments = attachments,
            .width = self->swapchain_extent.width,
            .height = self->swapchain_extent.height,
            .layers = 1,
        };

        if (vkCreateFramebuffer(self->device, &framebuffer_info, NULL,
                &self->framebuffers[i]) != VK_SUCCESS) {
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
        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    struct VkAttachmentDescription normal_attachment = {
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    struct VkAttachmentDescription albedo_attachment = {
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    struct VkAttachmentDescription offscreen_depth_attachment = {
        .format = depth_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
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
           self->device, &offscreen_render_pass_info, NULL, &self->offscreen_render_pass) !=
           VK_SUCCESS) fatal("Failed to create render pass.");

    create_attachment(&self->offscreen_depth, self->physical_device,
        self->device, self->swapchain_extent.width, self->swapchain_extent.height,
        depth_format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    create_attachment(&self->offscreen_position, self->physical_device,
        self->device, self->swapchain_extent.width, self->swapchain_extent.height,
        position_attachment.format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    create_attachment(&self->offscreen_albedo, self->physical_device,
        self->device, self->swapchain_extent.width, self->swapchain_extent.height,
        albedo_attachment.format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    create_attachment(&self->offscreen_normal, self->physical_device,
        self->device, self->swapchain_extent.width, self->swapchain_extent.height,
        normal_attachment.format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

    VkImageView offscreen_attachments[4] = {
        self->offscreen_position.view,
        self->offscreen_normal.view,
        self->offscreen_albedo.view,
        self->offscreen_depth.view,
    };
    VkFramebufferCreateInfo offscreen_framebuffer_info = {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .renderPass = self->offscreen_render_pass,
        .attachmentCount = 4,
        .pAttachments = offscreen_attachments,
        .width = self->swapchain_extent.width,
        .height = self->swapchain_extent.height,
        .layers = 1,
    };

    if (vkCreateFramebuffer(self->device, &offscreen_framebuffer_info, NULL,
            &self->offscreen_framebuffer) != VK_SUCCESS) {
        fatal("Failed to create framebuffer.");
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
        .width  = (float) self->swapchain_extent.width,
        .height = (float) self->swapchain_extent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const VkRect2D scissor = {
        .offset = {0, 0},
        .extent = self->swapchain_extent,
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
                    self->device, "./shaders/deferred.vert.spv");
    VkShaderModule deferred_fragment_shader = create_shader_module(
                self->device, "./shaders/deferred.frag.spv");

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
        .layout = self->graphics_pipeline_layout,
        .renderPass = self->render_pass,
        .subpass = 0,
    };

    if (vkCreateGraphicsPipelines(
                              self->device,
                              VK_NULL_HANDLE,
                              1,
                              &pipeline_info,
                              NULL,
                              &self->graphics_pipeline) != VK_SUCCESS) {
        fatal("Failed to create graphics pipeline.");
    }

    vkDestroyShaderModule(self->device, deferred_fragment_shader, NULL);
    vkDestroyShaderModule(self->device, deferred_vertex_shader, NULL);

    // G-buffer write pipeline
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;

    shader_stages[0].module = create_shader_module(self->device, "./shaders/mrt.vert.spv");
    shader_stages[1].module = create_shader_module(self->device, "./shaders/mrt.frag.spv");

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
    attribute_descriptions[1].offset = offsetof(Vertex, color);

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

    pipeline_info.renderPass = self->offscreen_render_pass;

    VkPipelineColorBlendAttachmentState blend_states[3] = {
        color_blend_attachment, color_blend_attachment, color_blend_attachment, 
    };
    color_blending.attachmentCount = 3;
    color_blending.pAttachments = blend_states;

    if (vkCreateGraphicsPipelines(
                          self->device,
                          VK_NULL_HANDLE,
                          1,
                          &pipeline_info,
                          NULL,
                          &self->offscreen_graphics_pipeline) != VK_SUCCESS) {
        fatal("Failed to create graphics pipeline.");
    }

    // CREATE COLOR IMAGE
    create_2d_image(self->physical_device, self->device, self->swapchain_extent.width,
            self->swapchain_extent.height, VK_SAMPLE_COUNT_1_BIT, self->swapchain_format,
            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &self->color_att.image,
            &self->color_att.memory);
    create_2d_image_view(self->device, self->color_att.image, self->swapchain_format, VK_IMAGE_ASPECT_COLOR_BIT,
            &self->color_att.view);

    // CREATE DESCRIPTOR POOLS
    // TODO rewrite to be more specific
    enum { descriptor_type_count = 1 };
    VkDescriptorType descriptor_types[descriptor_type_count] = {
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    };
    VkDescriptorPoolSize* pool_sizes = malloc_nofail(
            sizeof(VkDescriptorPoolSize) * descriptor_type_count);
    for (size_t i=0; i < descriptor_type_count; i++) {
        pool_sizes[i].type = descriptor_types[i];
        pool_sizes[i].descriptorCount = FRAMES_IN_FLIGHT;
    }
    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = descriptor_type_count,
        .pPoolSizes = pool_sizes,
        .maxSets = FRAMES_IN_FLIGHT,
    };
    if (vkCreateDescriptorPool(
            self->device, &pool_info, NULL, &self->descriptor_pool)
            != VK_SUCCESS) {
        fatal("Failed to create descriptor pool.");
    }
    mem_free(pool_sizes);

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
            self->device, &texture_pool_info, NULL, &self->texture_descriptor_pool)
            != VK_SUCCESS) {
        fatal("Failed to create texture descriptor pool.");
    }

    // CREATE UNIFORM BUFFERS
    for (uint32_t i=0; i < FRAMES_IN_FLIGHT; i++) {
        create_buffer(
                self->physical_device,
                self->device,
                sizeof(Uniform),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                &self->view_proj_buffers[i],
                &self->view_proj_buffers_memories[i]
        );
    }

    self->current_frame = 0;

    // CREATE DESCRIPTOR SETS
    VkDescriptorSetAllocateInfo desc_set_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = self->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &self->view_proj_set_layout,
    };
    for (size_t i=0; i < FRAMES_IN_FLIGHT; i++) {
        if (vkAllocateDescriptorSets(
                self->device, &desc_set_alloc_info, &self->view_proj_sets[i]
                ) != VK_SUCCESS) {
            fatal("Failed to allocate descriptor sets.");
        }
    }

    // ALLOCATE COMMAND BUFFERS
    // TODO change to one-frame command buffer
    VkCommandBufferAllocateInfo cmdbuf_allocate_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = self->graphics_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 2,
    };
    if (vkAllocateCommandBuffers(self->device, &cmdbuf_allocate_info,
            self->command_buffers) != VK_SUCCESS) {
        fatal("Failed to allocate command buffers.");
    }
}

void render_draw_frame(Render* self, vec3 cam_pos, vec3 cam_dir, vec3 cam_up) {
    size_t current_frame = self->current_frame;

    Uniform uniform;
    mat4 proj;
    glm_perspective(0.6,
        self->swapchain_extent.width /
        (float) self->swapchain_extent.height, 0.01, 1000.0, proj);
    proj[1][1] *= -1;
    mat4 view;
    glm_look(cam_pos, cam_dir, cam_up, view);
    glm_mat4_mul(proj, view, uniform.view_proj);

    upload_to_device_local_buffer(
            (void*) &uniform,
            sizeof(uniform),
            self->view_proj_buffers[current_frame],
            self->physical_device,
            self->device,
            self->graphics_queue,
            self->graphics_command_pool
    );

    uint32_t image_index;
    VkResult acquire_image_result =
        vkAcquireNextImageKHR(self->device, self->swapchain, UINT64_MAX,
                self->image_available_semaphores[current_frame], VK_NULL_HANDLE,
                &image_index);

    if (acquire_image_result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_swapchain(self);
        return;
    } else if (acquire_image_result != VK_SUCCESS &&
            acquire_image_result != VK_SUBOPTIMAL_KHR) {
        fatal("Failed to acquire swapchain image.");
    }

    vkWaitForFences(
            self->device, 1, &self->commands_executed_fences[current_frame],
            VK_TRUE, UINT64_MAX);
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
    };
    if (vkBeginCommandBuffer(self->command_buffers[current_frame], &begin_info) !=
            VK_SUCCESS) {
        fatal("Failed to begin recording command buffer.");
    }
    VkClearValue clear_values[2] = {
        { .color = {0.0f, 0.0f, 0.0f, 1.0f} },
        { .depthStencil = {1.0f, 0} },
    };
    VkRenderPassBeginInfo render_pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = self->render_pass,
        .framebuffer = self->framebuffers[current_frame],
        .renderArea.offset = {0, 0},
        .renderArea.extent = self->swapchain_extent,
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };

    vkCmdBeginRenderPass(self->command_buffers[current_frame], &render_pass_info,
                         VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(self->command_buffers[current_frame],
            VK_PIPELINE_BIND_POINT_GRAPHICS, self->graphics_pipeline);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(self->command_buffers[current_frame], 0, 1,
            &self->vertex_buffer, &offset);
    vkCmdBindIndexBuffer(
            self->command_buffers[current_frame], self->index_buffer, 0,
            VK_INDEX_TYPE_UINT16);
    vkCmdBindDescriptorSets(self->command_buffers[current_frame],
            VK_PIPELINE_BIND_POINT_GRAPHICS, self->graphics_pipeline_layout,
            0, 1, &self->view_proj_sets[current_frame], 0, NULL);

    // Draw the nodes
    for (size_t n=0; n < g_nodes_count; n++) {
        Mesh* mesh = g_nodes[n].mesh;
        if (!mesh) continue;

        mat4 transform;
        memcpy(transform, g_nodes[n].transform, sizeof(transform));
        Node* parent = g_nodes[n].parent;
        while (parent) {
            mat4 parent_transform;
            memcpy(parent_transform, parent->transform, sizeof(mat4));
            glm_mat4_mul(parent_transform, transform, transform);
            parent = parent->parent;
        }

        vkCmdPushConstants(
                self->command_buffers[current_frame],
                self->graphics_pipeline_layout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                sizeof(PushConstants),
                &transform);

        for (size_t p=0; p < mesh->primitives_count; p++) {
            Primitive* primitive = &mesh->primitives[p];
            vkCmdBindDescriptorSets(
                self->command_buffers[current_frame],
                VK_PIPELINE_BIND_POINT_GRAPHICS, self->graphics_pipeline_layout,
                1, 1, &primitive->texture, 0, NULL);
            vkCmdDrawIndexed(self->command_buffers[current_frame],
                primitive->index_count, 1, primitive->index_offset,
                primitive->vertex_offset, 0);
        }
    }

    vkCmdEndRenderPass(self->command_buffers[current_frame]);

    if (vkEndCommandBuffer(self->command_buffers[current_frame]) != VK_SUCCESS) {
        fatal("Failed to record command buffer.");
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
        recreate_swapchain(self);
    } else if (present_result != VK_SUCCESS) {
        fatal("Failed to present swapchain image.");
    }

    self->current_frame = (current_frame + 1) % 2;

    glfwPollEvents();
}

bool render_exit(Render* render) {
    return glfwWindowShouldClose(g_window);
}

static void load_texture(Render* self, void* buffer, size_t len,
        VkImage* image, VkImageView* view, VkDeviceMemory* memory,
        VkDescriptorSet* descriptor_set)
{
    // Load pixels
    int tex_width, tex_height, tex_channels;
    stbi_uc* pixels = stbi_load_from_memory(
        buffer, len, &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
    if (!pixels) fatal("Failed to load texture.");
    uint32_t image_size = tex_width * tex_height * 4;

    // Upload pixels to staging buffer
    VkDeviceMemory texture_staging_memory;
    VkBuffer texture_staging = upload_data_to_staging_buffer(
        self->physical_device, self->device, pixels, image_size,
        &texture_staging_memory);

    stbi_image_free(pixels);

    // Create texture image
    create_2d_image(
            self->physical_device, self->device, tex_width, tex_height,
            VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, memory);

    {
    // Transition image layout for upload
    VkCommandBuffer cmdbuf = begin_one_time_command_buffer(
            self->device, self->graphics_command_pool);
    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *image,
        .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .subresourceRange.baseMipLevel = 0,
        .subresourceRange.levelCount = 1,
        .subresourceRange.baseArrayLayer = 0,
        .subresourceRange.layerCount = 1,
    };
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);
    submit_one_time_command_buffer(
            self->device, self->graphics_queue, cmdbuf,
            self->graphics_command_pool);
    }

    {
    // Copy staging buffer to image
    VkCommandBuffer cmdbuf = begin_one_time_command_buffer(
            self->device, self->graphics_command_pool);
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
    vkCmdCopyBufferToImage(cmdbuf, texture_staging, *image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    submit_one_time_command_buffer(self->device, self->graphics_queue, cmdbuf,
            self->graphics_command_pool);

    vkDestroyBuffer(self->device, texture_staging, NULL);
    vkFreeMemory(self->device, texture_staging_memory, NULL);
    }
    {
    // Transition image layout for shader access
    VkCommandBuffer cmdbuf = begin_one_time_command_buffer(
            self->device, self->graphics_command_pool);
    VkImageMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *image,
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
            self->device, self->graphics_queue, cmdbuf,
            self->graphics_command_pool);
    }

    create_2d_image_view(
            self->device, *image, VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_ASPECT_COLOR_BIT, view);

    VkDescriptorSetAllocateInfo tex_set_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = self->texture_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &self->texture_set_layout,
    };
    if (vkAllocateDescriptorSets(
            self->device, &tex_set_alloc_info, descriptor_set) != VK_SUCCESS) {
        fatal("Failed to allocate descriptor sets.");
    }

    VkDescriptorImageInfo texture_info = {
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .imageView = *view,
        .sampler = self->texture_sampler,
    };
    VkWriteDescriptorSet texture_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = *descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .pImageInfo = &texture_info,
    };
    vkUpdateDescriptorSets(self->device, 1, &texture_write, 0, NULL);
}

void render_upload_map_mesh(Render* self)
{
    // LOAD GLTF
    cgltf_options gltf_options = {0};
    cgltf_data* gltf_data = NULL;
    cgltf_result gltf_result = cgltf_parse_file(
                            &gltf_options, "cube.glb", &gltf_data);
    if (gltf_result != cgltf_result_success) fatal("Failed to load GLTF.");
    gltf_result = cgltf_load_buffers(&gltf_options, gltf_data, "cube.glb");
    if (gltf_result != cgltf_result_success) fatal("Failed to load GLTF buffers.");
    
    // Load materials
    g_texture_count = gltf_data->materials_count;
    g_texture_images = malloc_nofail(sizeof(VkImage) * g_texture_count);
    g_texture_image_views = malloc_nofail(sizeof(VkImageView) * g_texture_count);
    g_texture_image_memories =
        malloc_nofail(sizeof(VkDeviceMemory) * g_texture_count);
    g_texture_descriptor_sets = malloc_nofail(
            sizeof(VkDescriptorSet) * g_texture_count);
    for (size_t i=0; i < g_texture_count; i++) {
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

        load_texture(self, image_data, image_size, &g_texture_images[i],
                &g_texture_image_views[i], &g_texture_image_memories[i],
                &g_texture_descriptor_sets[i]);
    }

    g_meshes = malloc_nofail(sizeof(Mesh) * gltf_data->meshes_count);
    g_meshes_count = gltf_data->meshes_count;

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
        Mesh* mesh = &g_meshes[i];
        mesh->primitives_count = gltf_mesh->primitives_count;
        mesh->primitives = malloc_nofail(
                        sizeof(Primitive) * mesh->primitives_count);
        // Primitives
        for (size_t p=0; p < gltf_mesh->primitives_count; p++) {
            cgltf_primitive* gltf_primitive = &gltf_mesh->primitives[p];

            // Material
            size_t material_index = (size_t)(((char*)gltf_primitive->material -
                    (char*) gltf_data->materials) / sizeof(cgltf_material));
            mesh->primitives[p].texture = g_texture_descriptor_sets[material_index];

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
    g_nodes_count = gltf_data->nodes_count;
    g_nodes = malloc_nofail(sizeof(Node) * g_nodes_count);

    bool camera_found = false;
    for (size_t n=0; n < g_nodes_count; n++) {
        Node* node = &g_nodes[n];
        cgltf_node* gltf_node = &gltf_nodes[n];

        mat4 transform = GLM_MAT4_IDENTITY_INIT;
        DBASSERT(!gltf_node->has_matrix);
        if (gltf_node->has_scale) glm_scale(transform, gltf_node->scale);
        if (gltf_node->has_rotation) {
            versor quat;
            memcpy(quat, gltf_node->rotation, sizeof(vec4));
            glm_quat_rotate(transform, quat, transform);
        }
        if (gltf_node->has_translation) {
            glm_translate(transform, gltf_node->translation);
        }
        memcpy(node->transform, transform, sizeof(mat4));

        node->mesh = NULL;
        if (gltf_node->mesh) {
            size_t mesh_index = (size_t) (((char*) gltf_node->mesh -
                        (char*) gltf_data->meshes) / sizeof(cgltf_mesh));
            node->mesh = &g_meshes[mesh_index];
        }

        node->parent = NULL;
        if (gltf_node->parent) {
            size_t parent_index = (size_t) (((char*) gltf_node->parent -
                        (char*) gltf_nodes) / sizeof(cgltf_node));
            node->parent = &g_nodes[parent_index];
        }

        node->children_count = gltf_node->children_count;
        node->children = malloc_nofail(sizeof(Node*) * node->children_count);
        for (size_t c=0; c < gltf_node->children_count; c++) {
            size_t child_index = (size_t) (((char*) gltf_node->children[c] -
                        (char*) gltf_data->nodes) / sizeof(cgltf_node));
            node->children[c] = &g_nodes[child_index];    
        }

        // Load camera
        // TODO REALLY load camera
        /*
        cgltf_camera* gltf_camera = gltf_node->camera;
        if (gltf_camera) {
            camera_found = true;
            DBASSERT(!gltf_node->parent);
            DBASSERT(gltf_camera->type == cgltf_camera_type_perspective);


            glm_mat4_print(camview,stdout);
        }
        */
    }
    //DBASSERT(camera_found);

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
    mem_free(vertices);
    mem_free(indices);

    cgltf_free(gltf_data);

    for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo buffer_info = {
            .buffer = self->view_proj_buffers[i],
            .offset = 0,
            .range = sizeof(Uniform),
        };

        VkWriteDescriptorSet uniform_write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = self->view_proj_sets[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &buffer_info,
        };

        vkUpdateDescriptorSets(self->device, 1, &uniform_write, 0, NULL);
    }
}

static void cleanup_swapchain(Render* self)
{
    vkDestroyDescriptorPool(self->device, self->descriptor_pool, NULL);

    for (size_t i=0; i < FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(
                self->device, self->image_available_semaphores[i], NULL);
        vkDestroySemaphore(
                self->device, self->draw_finished_semaphores[i], NULL);
        vkDestroyFence(self->device, self->commands_executed_fences[i], NULL);
    }

    for (size_t i=0; i < FRAMES_IN_FLIGHT; i++) {
        vkDestroyFramebuffer(self->device, self->framebuffers[i], NULL);
    }
    
    vkDestroyImageView(self->device, self->depth_att.view, NULL);
    vkDestroyImage(self->device, self->depth_att.image, NULL);
    vkFreeMemory(self->device, self->depth_att.memory, NULL);

    vkDestroyImageView(self->device, self->color_att.view, NULL);
    vkDestroyImage(self->device, self->color_att.image, NULL);
    vkFreeMemory(self->device, self->color_att.memory, NULL);

    vkDestroyPipeline(self->device, self->graphics_pipeline, NULL);
    vkDestroyPipelineLayout(self->device, self->graphics_pipeline_layout, NULL);

    vkDestroyRenderPass(self->device, self->render_pass, NULL);

    for (uint32_t i=0; i < FRAMES_IN_FLIGHT; i++) {
        vkDestroyImageView(self->device, self->swapchain_image_views[i], NULL);
    }

    for (size_t i=0; i < FRAMES_IN_FLIGHT; i++) {
        vkDestroyBuffer(self->device, self->view_proj_buffers[i], NULL);
        vkFreeMemory(self->device, self->view_proj_buffers_memories[i], NULL);
    }

    vkDestroySwapchainKHR(self->device, self->swapchain, NULL);
}

void render_destroy(Render* self)
{
    for (size_t i=0; i < g_nodes_count; i++) mem_free(g_nodes[i].children);
    mem_free(g_nodes);
    for (size_t i=0; i < g_meshes_count; i++) {
        mem_free(g_meshes[i].primitives);
    }
    mem_free(g_meshes);

    vkDeviceWaitIdle(self->device);
    cleanup_swapchain(self);

    vkDestroySampler(self->device, self->texture_sampler, NULL);
    vkDestroyDescriptorPool(self->device, self->texture_descriptor_pool, NULL);
    for (size_t i=0; i < g_texture_count; i++) {     
        vkDestroyImageView(self->device, g_texture_image_views[i], NULL);
        vkDestroyImage(self->device, g_texture_images[i], NULL);
        vkFreeMemory(self->device, g_texture_image_memories[i], NULL);
    } 
    mem_free(g_texture_image_views);
    mem_free(g_texture_images);
    mem_free(g_texture_image_memories);
    mem_free(g_texture_descriptor_sets);

    vkDestroyDescriptorSetLayout(self->device, self->view_proj_set_layout, NULL);
    vkDestroyDescriptorSetLayout(self->device, self->texture_set_layout, NULL);

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

    glfwDestroyWindow(g_window);
    glfwTerminate();

    mem_free(self);
}

static void create_2d_image(VkPhysicalDevice physical_device, VkDevice device, uint32_t width, uint32_t height,
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

    if (vkCreateImage(device, &image_create_info, NULL, image) != VK_SUCCESS) {
        fatal("Failed to create image.");
    }

    VkMemoryRequirements memory_requirements;
    vkGetImageMemoryRequirements(device, *image, &memory_requirements);

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

    if (vkAllocateMemory(device, &allocate_info, NULL, memory)
            != VK_SUCCESS) fatal("Failed to allocate image memory.");

    vkBindImageMemory(device, *image, *memory, 0);
}

static void create_2d_image_view(VkDevice device, VkImage image, VkFormat format,
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
    if (vkCreateImageView(device, &create_info, NULL, image_view) != VK_SUCCESS) {
        fatal("Failed to create image views.");
    }
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
    if (!found) fatal("Failed to find depth format.");
    return depth_format;
}

static VkShaderModule create_shader_module(
                    VkDevice device, const char* path)
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
            device, &create_info, NULL, &shader_module) !=
            VK_SUCCESS) {
        fatal("Failed to create shader module.");
    }

    mem_free(shader_code);
    return shader_module;
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

static void submit_one_time_command_buffer(VkDevice device, VkQueue queue,
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

// TODO straighten out shady return values
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
            != VK_SUCCESS) return 2;

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
    submit_one_time_command_buffer(device, queue, command_buffer, command_pool);
}

static VkBuffer upload_data_to_staging_buffer(
    VkPhysicalDevice physical_device, VkDevice device, void* data,
    size_t size, VkDeviceMemory* staging_buffer_memory)
{
    VkDeviceSize device_size = size;

    VkBuffer staging_buffer;
    create_buffer(
            physical_device,
            device,
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            // TODO use non-coherent memory with a flush
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &staging_buffer,
            staging_buffer_memory
    );

    void *staging_buffer_mapped;
    vkMapMemory(
            device, *staging_buffer_memory, 0, device_size, 0,
            &staging_buffer_mapped);
    memcpy(staging_buffer_mapped, data, size);
    vkUnmapMemory(device, *staging_buffer_memory);
    return staging_buffer;
}

static void upload_to_device_local_buffer(
        void* data,
        size_t size,
        VkBuffer destination,
        VkPhysicalDevice physical_device,
        VkDevice device,
        VkQueue queue,
        VkCommandPool command_pool)
{
    VkDeviceSize device_size = size;
    VkDeviceMemory staging_buffer_memory;
    VkBuffer staging_buffer = upload_data_to_staging_buffer(
                physical_device, device, data, size, &staging_buffer_memory);
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

static void recreate_swapchain(Render* self)
{
    int width = 0, height = 0;
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(g_window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(self->device);

    cleanup_swapchain(self);

    render_swapchain_dependent_init(self);
}
