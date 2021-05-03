#ifndef VKHELPERS_H
#define VKHELPERS_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

typedef struct Buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
} Buffer;
void destroy_buffer(Buffer* buffer);

typedef struct Texture {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    VkDescriptorSet desc_set;
    int width;
    int height;
} Texture;
void destroy_texture(Texture* texture);

int find_memory_type(
        VkMemoryRequirements memory_requirements,
        VkMemoryPropertyFlags required_properties);
void create_2d_image(uint32_t width, uint32_t height,
        VkSampleCountFlagBits samples, VkFormat format, VkImageTiling tiling,
        VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
        VkImage* image, VkDeviceMemory* memory);
void create_2d_image_view(VkImage image, VkFormat format,
        VkImageAspectFlags aspect_flags, VkImageView* image_view);
VkFormat find_depth_format();
VkShaderModule create_shader_module(const char* path);
VkCommandBuffer begin_one_time_command_buffer(VkCommandPool command_pool);
void submit_one_time_command_buffer(VkQueue queue,
        VkCommandBuffer command_buffer, VkCommandPool command_pool);
int create_buffer(
        size_t size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, Buffer* buffer);
void copy_buffer(
        VkQueue queue,
        VkCommandPool command_pool,
        VkBuffer src_buffer, VkBuffer dst_buffer,
        VkDeviceSize device_size);
Buffer upload_data_to_staging_buffer(void* data, size_t size);
void upload_to_device_local_buffer(
        void* data,
        size_t size,
        Buffer* destination,
        VkQueue queue,
        VkCommandPool command_pool);
void device_local_buffer_from_data(
        void* data,
        size_t size,
        VkBufferUsageFlags usage,
        VkQueue queue,
        VkCommandPool command_pool,
        Buffer* buffer);

struct VkSubpassDependency default_start_dependency();
struct VkSubpassDependency default_end_dependency();
VkViewport default_viewport(float width, float height);
VkRect2D default_scissor(VkExtent2D extent);
VkPipelineViewportStateCreateInfo default_viewport_state(
        const VkViewport* p_viewport, const VkRect2D* p_scissor);
VkPipelineColorBlendAttachmentState default_color_blend_attachment_state();
VkPipelineDepthStencilStateCreateInfo default_depth_stencil_state();
VkPipelineRasterizationStateCreateInfo default_rasterizer(VkCullModeFlags cull_mode);
VkPipelineShaderStageCreateInfo shader_stage_info(
        VkShaderStageFlagBits stage, VkShaderModule module);

#endif
