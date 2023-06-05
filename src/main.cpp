#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

#ifdef NDEBUG
const bool VK_ENABLE_VALIDATION_LAYERS = false;
#else
const bool VK_ENABLE_VALIDATION_LAYERS = true;
#endif

const std::vector<const char*> VK_VALIDATION_LAYERS = {
    "VK_LAYER_KHRONOS_validation",
};

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

    if (VK_ENABLE_VALIDATION_LAYERS) {
      std::vector<VkLayerProperties> vulkanLayers;
      readVulkanSupportedLayers(vulkanLayers);
      logVulkanSupportedLayers(vulkanLayers);
      checkSupportsVulkanValidationLayer(VK_VALIDATION_LAYERS, vulkanLayers);
    }

    std::vector<const char*> requiredExtensions;
    readGlfwRequestedExtensions(requiredExtensions);

    VkInstanceCreateInfo createInfo{};
    setupVulkanInstanceCreateInfo(createInfo, appInfo, requiredExtensions,
                                  VK_ENABLE_VALIDATION_LAYERS,
                                  VK_VALIDATION_LAYERS);

    std::vector<VkExtensionProperties> vulkanExtensions;
    readVulkanSupportedExtensions(vulkanExtensions);
    logVulkanSupportedExtensions(vulkanExtensions);
    checkSupportsRequiredExtensions(requiredExtensions, vulkanExtensions);

    if (vkCreateInstance(&createInfo, nullptr, &vkInstance) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create Vulkan instance");
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

  void setupVulkanInstanceCreateInfo(
      VkInstanceCreateInfo& createInfo, VkApplicationInfo& appInfo,
      std::vector<const char*>& requiredExtensions, bool enableValidationLayers,
      const std::vector<const char*>& validationLayers) {
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(requiredExtensions.size());
    createInfo.ppEnabledExtensionNames = requiredExtensions.data();

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }
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
    spdlog::debug("Available VK extensions ({} total):",
                  vulkanExtensions.size());
    for (const auto& extension : vulkanExtensions) {
      spdlog::debug("\t{}", extension.extensionName);
    }
  }

  void readGlfwRequestedExtensions(
      std::vector<const char*>& requiredExtensions) {
    uint32_t numGlfwExtensions = 0;
    const char** glfwExtensions =
        glfwGetRequiredInstanceExtensions(&numGlfwExtensions);

    for (uint32_t i = 0; i < numGlfwExtensions; i++) {
      requiredExtensions.push_back(glfwExtensions[i]);
    }
  }

  void checkSupportsRequiredExtensions(
      const std::vector<const char*>& requiredExtensions,
      const std::vector<VkExtensionProperties>& vulkanExtensions) {
    for (const auto& glfwExtension : requiredExtensions) {
      bool found = false;
      for (const auto& vulkanExtension : vulkanExtensions) {
        if (strcmp(glfwExtension, vulkanExtension.extensionName) == 0) {
          found = true;
          break;
        }
      }

      if (!found) {
        throw std::runtime_error(std::format(
            "Extension {} required by GLFW is not supported by Vulkan",
            glfwExtension));
      }
    }
  }

  void readVulkanSupportedLayers(std::vector<VkLayerProperties>& vulkanLayers) {
    uint32_t numVulkanLayers;
    vkEnumerateInstanceLayerProperties(&numVulkanLayers, nullptr);

    vulkanLayers.resize(numVulkanLayers);
    vkEnumerateInstanceLayerProperties(&numVulkanLayers, vulkanLayers.data());
  }

  void logVulkanSupportedLayers(
      const std::vector<VkLayerProperties>& vulkanLayers) {
    spdlog::debug("Available VK layers ({} total):", vulkanLayers.size());
    for (const auto& layer : vulkanLayers) {
      spdlog::debug("\t{}", layer.layerName);
    }
  }

  void checkSupportsVulkanValidationLayer(
      const std::vector<const char*>& requiredLayers,
      const std::vector<VkLayerProperties>& vulkanLayers) {
    for (const char* layer : requiredLayers) {
      bool found = false;
      for (const auto& vulkanLayer : vulkanLayers) {
        if (strcmp(layer, vulkanLayer.layerName) == 0) {
          found = true;
          break;
        }
      }

      if (!found) {
        throw std::runtime_error(std::format(
            "Validation layer {} requested, but not available", layer));
      }
    }
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(glfwWindow)) {
      glfwPollEvents();
    }
  }

  void cleanup() {
    cleanupVulkan();
    cleanupGlfw();
  }

  void cleanupGlfw() {
    glfwDestroyWindow(glfwWindow);
    glfwTerminate();
  }

  void cleanupVulkan() { vkDestroyInstance(vkInstance, nullptr); }
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
