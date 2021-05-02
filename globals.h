// Global variables accessible across files

#ifndef GLOBALS_H
#define GLOBALS_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
extern GLFWwindow* g_window;
extern VkDevice g_device;
extern VkPhysicalDevice g_physical_device;

#endif
