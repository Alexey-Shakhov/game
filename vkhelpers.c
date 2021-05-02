#include "vkhelpers.h"
#include <string.h>
#include <stdbool.h>
#include "globals.h"
#include "utils.h"
#include "alloc.h"

void destroy_buffer(Buffer* buffer)
{
    vkDestroyBuffer(g_device, buffer->buffer, NULL);
    vkFreeMemory(g_device, buffer->memory, NULL);
}

int find_memory_type(
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

void create_2d_image(uint32_t width, uint32_t height,
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
            properties
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

void create_2d_image_view(VkImage image, VkFormat format,
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

VkFormat find_depth_format() {
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

VkShaderModule create_shader_module(const char* path)
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

VkCommandBuffer begin_one_time_command_buffer(VkCommandPool command_pool)
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

void submit_one_time_command_buffer(VkQueue queue,
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

// TODO straighten out shady return values
int create_buffer(
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

void copy_buffer(
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

Buffer upload_data_to_staging_buffer(void* data, size_t size)
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

void upload_to_device_local_buffer(
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

void device_local_buffer_from_data(
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
