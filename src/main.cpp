#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

class HelloTriangleApplication {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
  GLFWwindow* glfwWindow;
  VkInstance vkInstance;

  void initWindow() {
    auto result = glfwInit();
    if (result != GLFW_TRUE) {
      throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    glfwWindow = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }

  void initVulkan() { createVulkanInstance(); }

  void createVulkanInstance() {
    VkApplicationInfo appInfo{};
    setupVulkanApplicationInfo(appInfo);

    VkInstanceCreateInfo createInfo{};
    uint32_t numGlfwExtensions = 0;
    const char** glfwExtensions = nullptr;
    setupVulkanInstanceCreateInfo(createInfo, appInfo, glfwExtensions,
                                  numGlfwExtensions);

    std::vector<VkExtensionProperties> vulkanExtensions;
    readVulkanSupportedExtensions(vulkanExtensions);
    logVulkanSupportedExtensions(vulkanExtensions);
    checkSupportsGlfwRequiredExtensions(numGlfwExtensions, glfwExtensions,
                                        vulkanExtensions);

    if (vkCreateInstance(&createInfo, nullptr, &vkInstance) != VK_SUCCESS) {
      throw new std::runtime_error("Failed to create Vulkan instance");
    }
  }

  void setupVulkanApplicationInfo(VkApplicationInfo& appInfo) {
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
  }

  void setupVulkanInstanceCreateInfo(VkInstanceCreateInfo& createInfo,
                                     VkApplicationInfo& appInfo,
                                     const char**& glfwExtensions,
                                     uint32_t& numGlfwExtensions) {
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&numGlfwExtensions);
    createInfo.enabledExtensionCount = numGlfwExtensions;
    createInfo.ppEnabledExtensionNames = glfwExtensions;
    createInfo.enabledLayerCount = 0;
  }

  void readVulkanSupportedExtensions(
      std::vector<VkExtensionProperties>& vulkanExtensions) {
    uint32_t numVulkanExtensions = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &numVulkanExtensions,
                                           nullptr);
    vulkanExtensions.resize(numVulkanExtensions);
    vkEnumerateInstanceExtensionProperties(nullptr, &numVulkanExtensions,
                                           vulkanExtensions.data());
  }

  void logVulkanSupportedExtensions(
      const std::vector<VkExtensionProperties>& vulkanExtensions) {
    spdlog::debug("Available extensions:");
    for (const auto& extension : vulkanExtensions) {
      spdlog::debug("\t{}", extension.extensionName);
    }
  }

  void checkSupportsGlfwRequiredExtensions(
      const uint32_t& numGlfwExtensions, const char** glfwExtensions,
      std::vector<VkExtensionProperties>& vulkanExtensions) {
    for (uint32_t i = 0; i < numGlfwExtensions; i++) {
      auto glfwExtension = glfwExtensions[i];

      bool found = false;
      for (const auto& vulkanExtension : vulkanExtensions) {
        if (strcmp(glfwExtension, vulkanExtension.extensionName) == 0) {
          found = true;
          break;
        }
      }

      if (!found) {
        throw new std::runtime_error(std::format(
            "Extension {} required by GLFW is not supported by Vulkan",
            glfwExtension));
      }
    }
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(glfwWindow)) {
      glfwPollEvents();
    }
  }

  void cleanup() {
    glfwDestroyWindow(glfwWindow);
    glfwTerminate();
  }
};

int main() {
  HelloTriangleApplication app;

  spdlog::set_level(spdlog::level::debug);

  try {
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
