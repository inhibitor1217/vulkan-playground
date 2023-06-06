#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <iostream>
#include <optional>
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
  GLFWwindow* glfwWindow = nullptr;

  VkInstance vkInstance = VK_NULL_HANDLE;
  VkPhysicalDevice vkPhysicalDevice = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT vkDebugMessenger = VK_NULL_HANDLE;

  struct VkPhysicalDeviceQueueFamilies {
    std::optional<uint32_t> graphicsFamily;

    bool isOk() { return graphicsFamily.has_value(); }
  };

  void initWindow() {
    auto result = glfwInit();
    if (result != GLFW_TRUE) {
      throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    glfwWindow = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }

  void initVulkan() {
    createVulkanInstance();
    if (VK_ENABLE_VALIDATION_LAYERS) {
      createVulkanDebugMessenger();
    }
    createVulkanPhysicalDevice();
  }

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
    requireGlfwRequestedExtensions(requiredExtensions);
    if (VK_ENABLE_VALIDATION_LAYERS) {
      requireValidationRequestedExtensions(requiredExtensions);
    }

    std::vector<VkExtensionProperties> vulkanExtensions;
    readVulkanSupportedExtensions(vulkanExtensions);
    logVulkanSupportedExtensions(vulkanExtensions);
    checkSupportsRequiredExtensions(requiredExtensions, vulkanExtensions);

    VkInstanceCreateInfo createInfo{};
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    setupVulkanInstanceCreateInfo(
        createInfo, debugCreateInfo, appInfo, requiredExtensions,
        VK_ENABLE_VALIDATION_LAYERS, VK_VALIDATION_LAYERS);

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
      VkInstanceCreateInfo& createInfo,
      VkDebugUtilsMessengerCreateInfoEXT& debugCreateInfo,
      VkApplicationInfo& appInfo, std::vector<const char*>& requiredExtensions,
      bool enableValidationLayers,
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

    if (enableValidationLayers) {
      setupVulkanDebugMessengerCreateInfo(debugCreateInfo);
      createInfo.pNext = &debugCreateInfo;
    } else {
      createInfo.pNext = nullptr;
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

  void requireGlfwRequestedExtensions(
      std::vector<const char*>& requiredExtensions) {
    uint32_t numGlfwExtensions = 0;
    const char** glfwExtensions =
        glfwGetRequiredInstanceExtensions(&numGlfwExtensions);

    for (uint32_t i = 0; i < numGlfwExtensions; i++) {
      requiredExtensions.push_back(glfwExtensions[i]);
    }
  }

  void requireValidationRequestedExtensions(
      std::vector<const char*>& requiredExtensions) {
    requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
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

  void createVulkanDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    setupVulkanDebugMessengerCreateInfo(createInfo);

    auto createDebugUtilsMessengerEXT =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            vkInstance, "vkCreateDebugUtilsMessengerEXT");
    if (createDebugUtilsMessengerEXT == nullptr) {
      throw std::runtime_error(
          "Failed to setup debug messenger: vkCreateDebugUtilsMessengerEXT not "
          "found");
    }

    if (createDebugUtilsMessengerEXT(vkInstance, &createInfo, nullptr,
                                     &vkDebugMessenger) != VK_SUCCESS) {
      throw std::runtime_error("Failed to setup debug messenger");
    }
  }

  void setupVulkanDebugMessengerCreateInfo(
      VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = vkDebugMessengerCallback;
    createInfo.pUserData = nullptr;
  }

  void createVulkanPhysicalDevice() {
    std::vector<VkPhysicalDevice> devices;

    readAvailableVulkanPhysicalDevices(devices);
    logAvailableVulkanPhysicalDevices(devices);

    auto it = std::find_if(
        devices.begin(), devices.end(), [this](const VkPhysicalDevice& device) {
          return this->isVulkanPhysicalDeviceSuitable(device);
        });

    if (it == devices.end()) {
      vkPhysicalDevice = VK_NULL_HANDLE;
    } else {
      vkPhysicalDevice = *it;
    }

    if (vkPhysicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("Failed to find suitable physical device");
    }
  }

  void readAvailableVulkanPhysicalDevices(
      std::vector<VkPhysicalDevice>& devices) {
    uint32_t numDevices = 0;
    vkEnumeratePhysicalDevices(vkInstance, &numDevices, nullptr);

    if (numDevices == 0) {
      devices.clear();
      return;
    }

    devices.resize(numDevices);
    vkEnumeratePhysicalDevices(vkInstance, &numDevices, devices.data());
  }

  void logAvailableVulkanPhysicalDevices(
      const std::vector<VkPhysicalDevice>& devices) {
    spdlog::debug("Available VK physical devices ({} total):", devices.size());
    for (const auto& device : devices) {
      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(device, &deviceProperties);
      spdlog::debug("\t{}", deviceProperties.deviceName);
    }
  }

  bool isVulkanPhysicalDeviceSuitable(VkPhysicalDevice device) {
    VkPhysicalDeviceQueueFamilies queueFamilies;
    readVulkanPhysicalDeviceQueueFamilyProperties(device, queueFamilies);

    return queueFamilies.isOk();
  }

  void readVulkanPhysicalDeviceQueueFamilyProperties(
      VkPhysicalDevice device, VkPhysicalDeviceQueueFamilies& queueFamilies) {
    uint32_t numQueueFamilies = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &numQueueFamilies,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilyProperties(
        numQueueFamilies);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &numQueueFamilies,
                                             queueFamilyProperties.data());

    auto it =
        std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
                     [](const VkQueueFamilyProperties& queueFamily) {
                       return queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT;
                     });
    if (it != queueFamilyProperties.end()) {
      queueFamilies.graphicsFamily = static_cast<uint32_t>(
          std::distance(queueFamilyProperties.begin(), it));
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

  void cleanupVulkan() {
    if (VK_ENABLE_VALIDATION_LAYERS) {
      auto destroyDebugUtilsMessengerEXT =
          (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
              vkInstance, "vkDestroyDebugUtilsMessengerEXT");
      if (destroyDebugUtilsMessengerEXT != nullptr) {
        destroyDebugUtilsMessengerEXT(vkInstance, vkDebugMessenger, nullptr);
      }
    }

    vkDestroyInstance(vkInstance, nullptr);
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL vkDebugMessengerCallback(
      VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
      VkDebugUtilsMessageTypeFlagsEXT messageType,
      const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
      void* pUserData) {
    if (pCallbackData == nullptr) {
      return VK_FALSE;
    }

    const char* message = pCallbackData->pMessage;

    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
      spdlog::error(message);
    } else if (messageSeverity &
               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
      spdlog::warn(message);
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
      spdlog::info(message);
    } else if (messageSeverity &
               VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
      spdlog::debug(message);
    }

    return VK_FALSE;
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
