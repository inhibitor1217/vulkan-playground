#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;

  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription{};

    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

    // position
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    // color
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    return attributeDescriptions;
  }
};

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<uint16_t> indices;

  size_t numVertices() const { return vertices.size(); }
  size_t numIndices() const { return indices.size(); }
  size_t vertexBufferSize() const { return numVertices() * sizeof(Vertex); }
  size_t indexBufferSize() const { return numIndices() * sizeof(uint16_t); }
};

const Mesh mesh = {{{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
                    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
                    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}},
                   {0, 1, 2, 2, 3, 0}};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

class Application {
 public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

 private:
#ifdef NDEBUG
  const bool VK_ENABLE_VALIDATION_LAYERS = false;
#else
  const bool VK_ENABLE_VALIDATION_LAYERS = true;
#endif

  const std::vector<const char*> VK_VALIDATION_LAYERS = {
      "VK_LAYER_KHRONOS_validation",
  };

  const std::vector<const char*> VK_DEVICE_EXTENSIONS = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  };

  const int MAX_FRAMES_IN_FLIGHT = 2;

  struct VkPhysicalDeviceQueueFamilies {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isOk() {
      return graphicsFamily.has_value() && presentFamily.has_value();
    }
  };

  struct VkSwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;

    bool isOk() { return !formats.empty() && !presentModes.empty(); }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat() {
      for (const auto& surfaceFormat : formats) {
        if (surfaceFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
            surfaceFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
          return surfaceFormat;
        }
      }

      return formats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode() {
      return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(GLFWwindow* glfwWindow) {
      // Just use the window's current extent
      if (capabilities.currentExtent.width !=
          std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
      } else {
        // In this case, we need to pick the extent that best matches the window
        int width, height;
        glfwGetFramebufferSize(glfwWindow, &width, &height);

        VkExtent2D actualExtent = {
            std::clamp(static_cast<uint32_t>(width),
                       capabilities.minImageExtent.width,
                       capabilities.maxImageExtent.width),
            std::clamp(static_cast<uint32_t>(height),
                       capabilities.minImageExtent.height,
                       capabilities.maxImageExtent.height),
        };

        return actualExtent;
      }
    }
  };

  struct VkFrameRenderResources {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;
    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBefferMemory;
    void* uniformBufferMapped;
  };

  GLFWwindow* glfwWindow = nullptr;

  VkInstance vkInstance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT vkDebugMessenger = VK_NULL_HANDLE;

  VkPhysicalDevice vkPhysicalDevice = VK_NULL_HANDLE;
  VkDevice vkDevice = VK_NULL_HANDLE;
  VkQueue vkGraphicsQueue = VK_NULL_HANDLE;
  VkQueue vkPresentQueue = VK_NULL_HANDLE;
  VkSurfaceKHR vkSurface = VK_NULL_HANDLE;

  VkSwapchainKHR vkSwapchain = VK_NULL_HANDLE;
  std::vector<VkImage> vkSwapchainImages;
  VkFormat vkSwapchainFormat = VK_FORMAT_UNDEFINED;
  VkExtent2D vkSwapchainExtent = {0, 0};
  std::vector<VkImageView> vkSwapchainImageViews;
  std::vector<VkFramebuffer> vkSwapchainFramebuffers;

  VkRenderPass vkRenderPass = VK_NULL_HANDLE;
  VkDescriptorSetLayout vkDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout vkPipelineLayout = VK_NULL_HANDLE;
  VkPipeline vkGraphicsPipeline = VK_NULL_HANDLE;

  VkCommandPool vkCommandPool = VK_NULL_HANDLE;
  VkDescriptorPool vkDescriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> vkDescriptorSets;

  std::vector<VkFrameRenderResources> vkFrameRenderResources;
  uint32_t currentFrame = 0;

  VkBuffer vkVertexBuffer;
  VkDeviceMemory vkVertexBufferMemory;
  VkBuffer vkIndexBuffer;
  VkDeviceMemory vkIndexBufferMemory;

  void initWindow() {
    auto result = glfwInit();
    if (result != GLFW_TRUE) {
      throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfwWindow = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }

  void initVulkan() {
    createVulkanInstance();
    if (VK_ENABLE_VALIDATION_LAYERS) {
      createVulkanDebugMessenger();
    }
    createVulkanSurface();
    createVulkanPhysicalDevice();
    createVulkanLogicalDevice();
    createVulkanSwapchain();
    createVulkanImageViews();
    createVulkanRenderPass();
    createVulkanDescriptorSetLayout();
    createVulkanGraphicsPipeline();
    createVulkanFramebuffers();
    createVulkanCommandPool();
    createVulkanVertexBuffer();
    createVulkanIndexBuffer();
    createVulkanFrameRenderResources();
    createVulkanDescriptorPool();
    createVulkanDescriptorSets();
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

  void createVulkanSurface() {
    if (glfwCreateWindowSurface(vkInstance, glfwWindow, nullptr, &vkSurface) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create window surface");
    }
  }

  void createVulkanPhysicalDevice() {
    std::vector<VkPhysicalDevice> devices;

    readAvailableVulkanPhysicalDevices(devices);
    logAvailableVulkanPhysicalDevices(devices);

    auto it = std::find_if(devices.begin(), devices.end(),
                           [this](const VkPhysicalDevice& device) {
                             return this->isVulkanPhysicalDeviceSuitable(
                                 device, VK_DEVICE_EXTENSIONS);
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

  bool isVulkanPhysicalDeviceSuitable(
      VkPhysicalDevice device,
      const std::vector<const char*>& requiredDeviceExtensions) {
    VkPhysicalDeviceQueueFamilies queueFamilies;
    readVulkanPhysicalDeviceQueueFamilyProperties(device, queueFamilies);

    return queueFamilies.isOk() &&
           checkSupportsRequiredDeviceExtension(device,
                                                requiredDeviceExtensions) &&
           checkSupportsSwapChain(device);
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

    for (uint32_t i = 0; i < numQueueFamilies; i++) {
      auto queueFamily = queueFamilyProperties[i];

      if (queueFamily.queueCount > 0 &&
          queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        queueFamilies.graphicsFamily = i;
      }

      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, vkSurface,
                                           &presentSupport);
      if (presentSupport) {
        queueFamilies.presentFamily = i;
      }
    }
  }

  bool checkSupportsRequiredDeviceExtension(
      VkPhysicalDevice device,
      const std::vector<const char*>& requiredDeviceExtensions) {
    uint32_t numDeviceExtensions = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &numDeviceExtensions,
                                         nullptr);

    std::vector<VkExtensionProperties> deviceExtensions(numDeviceExtensions);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &numDeviceExtensions,
                                         deviceExtensions.data());

    return std::all_of(
        requiredDeviceExtensions.begin(), requiredDeviceExtensions.end(),
        [&deviceExtensions](const char* requiredDeviceExtension) {
          return std::find_if(
                     deviceExtensions.begin(), deviceExtensions.end(),
                     [&requiredDeviceExtension](
                         const VkExtensionProperties& deviceExtension) {
                       return strcmp(deviceExtension.extensionName,
                                     requiredDeviceExtension) == 0;
                     }) != deviceExtensions.end();
        });
  }

  bool checkSupportsSwapChain(VkPhysicalDevice device) {
    VkSwapChainSupportDetails swapChainSupport =
        readVulkanSwapChainSupport(device);
    return swapChainSupport.isOk();
  }

  VkSwapChainSupportDetails readVulkanSwapChainSupport(
      VkPhysicalDevice device) {
    VkSwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, vkSurface,
                                              &details.capabilities);

    uint32_t numFormats;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, vkSurface, &numFormats,
                                         nullptr);
    details.formats.resize(numFormats);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, vkSurface, &numFormats,
                                         details.formats.data());

    uint32_t numPresentModes;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, vkSurface,
                                              &numPresentModes, nullptr);
    details.presentModes.resize(numPresentModes);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, vkSurface, &numPresentModes, details.presentModes.data());

    return details;
  }

  void createVulkanLogicalDevice() {
    VkPhysicalDeviceQueueFamilies queueFamilies;
    readVulkanPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice,
                                                  queueFamilies);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    setupVulkanDeviceQueueCreateInfo(queueFamilies, queueCreateInfos);

    VkPhysicalDeviceFeatures deviceFeatures{};
    setupVulkanPhysicalDeviceFeatures(deviceFeatures);

    VkDeviceCreateInfo createInfo{};
    setupVulkanDeviceCreateInfo(
        createInfo, queueCreateInfos, VK_DEVICE_EXTENSIONS, deviceFeatures,
        VK_ENABLE_VALIDATION_LAYERS, VK_VALIDATION_LAYERS);

    if (vkCreateDevice(vkPhysicalDevice, &createInfo, nullptr, &vkDevice) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create logical device");
    }

    vkGetDeviceQueue(vkDevice, queueFamilies.graphicsFamily.value(), 0,
                     &vkGraphicsQueue);
    vkGetDeviceQueue(vkDevice, queueFamilies.presentFamily.value(), 0,
                     &vkPresentQueue);
  }

  void setupVulkanDeviceQueueCreateInfo(
      const VkPhysicalDeviceQueueFamilies& queueFamilies,
      std::vector<VkDeviceQueueCreateInfo>& queueCreateInfos) {
    float queuePriority = 1.0f;

    std::set<uint32_t> uniqueQueueFamilies = {
        queueFamilies.graphicsFamily.value(),
        queueFamilies.presentFamily.value(),
    };

    for (uint32_t queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo{};

      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;

      queueCreateInfos.push_back(queueCreateInfo);
    }
  }

  void setupVulkanPhysicalDeviceFeatures(
      VkPhysicalDeviceFeatures& deviceFeatures) {
    // No features required for now.
  }

  void setupVulkanDeviceCreateInfo(
      VkDeviceCreateInfo& createInfo,
      const std::vector<VkDeviceQueueCreateInfo>& queueCreateInfos,
      const std::vector<const char*>& deviceExtensions,
      const VkPhysicalDeviceFeatures& deviceFeatures,
      bool enableValidationLayers,
      const std::vector<const char*>& validationLayers) {
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }
  }

  void createVulkanSwapchain() {
    auto details = readVulkanSwapChainSupport(vkPhysicalDevice);

    auto surfaceFormat = details.chooseSwapSurfaceFormat();
    auto presentMode = details.chooseSwapPresentMode();
    auto extent = details.chooseSwapExtent(glfwWindow);
    auto imageCount = details.capabilities.minImageCount + 1;
    if (details.capabilities.maxImageCount > 0) {
      imageCount = std::min(imageCount, details.capabilities.maxImageCount);
    }

    VkSwapchainCreateInfoKHR createInfo{};

    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = vkSurface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkPhysicalDeviceQueueFamilies queueFamilies;
    readVulkanPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice,
                                                  queueFamilies);
    if (queueFamilies.graphicsFamily != queueFamilies.presentFamily) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      uint32_t queueFamilyIndices[] = {queueFamilies.graphicsFamily.value(),
                                       queueFamilies.presentFamily.value()};
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      createInfo.queueFamilyIndexCount = 0;
      createInfo.pQueueFamilyIndices = nullptr;
    }

    createInfo.preTransform = details.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(vkDevice, &createInfo, nullptr, &vkSwapchain) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create swap chain");
    }

    uint32_t numSwapchainImages = 0;
    vkGetSwapchainImagesKHR(vkDevice, vkSwapchain, &numSwapchainImages,
                            nullptr);
    vkSwapchainImages.resize(numSwapchainImages);
    vkGetSwapchainImagesKHR(vkDevice, vkSwapchain, &numSwapchainImages,
                            vkSwapchainImages.data());

    vkSwapchainFormat = surfaceFormat.format;
    vkSwapchainExtent = extent;
  }

  void createVulkanImageViews() {
    vkSwapchainImageViews.resize(vkSwapchainImages.size());

    for (int i = 0; i < vkSwapchainImageViews.size(); i++) {
      auto image = vkSwapchainImages[i];
      VkImageViewCreateInfo createInfo{};

      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = image;
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      createInfo.format = vkSwapchainFormat;
      createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;

      if (vkCreateImageView(vkDevice, &createInfo, nullptr,
                            &vkSwapchainImageViews[i]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image views");
      }
    }
  }

  void createVulkanRenderPass() {
    VkAttachmentDescription colorAttachment{};

    colorAttachment.format = vkSwapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};

    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};

    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};

    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};

    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(vkDevice, &renderPassInfo, nullptr, &vkRenderPass) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create render pass");
    }
  }

  void createVulkanDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding{};

    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};

    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    if (vkCreateDescriptorSetLayout(vkDevice, &layoutInfo, nullptr,
                                    &vkDescriptorSetLayout) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create descriptor set layout");
    }
  }

  void createVulkanGraphicsPipeline() {
    auto vertexShader = readFile("res/triangle.vert.spv");
    auto fragmentShader = readFile("res/triangle.frag.spv");

    auto vertexShaderModule = createVulkanShaderModule(vertexShader);
    auto fragmentShaderModule = createVulkanShaderModule(fragmentShader);

    VkPipelineShaderStageCreateInfo vertexShaderStageInfo{};

    vertexShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertexShaderStageInfo.module = vertexShaderModule;
    vertexShaderStageInfo.pName = "main";
    vertexShaderStageInfo.pSpecializationInfo = nullptr;

    VkPipelineShaderStageCreateInfo fragmentShaderStageInfo{};

    fragmentShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragmentShaderStageInfo.module = fragmentShaderModule;
    fragmentShaderStageInfo.pName = "main";
    fragmentShaderStageInfo.pSpecializationInfo = nullptr;

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertexShaderStageInfo,
                                                      fragmentShaderStageInfo};

    std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                 VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamicStateInfo{};

    dynamicStateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateInfo.dynamicStateCount =
        static_cast<uint32_t>(dynamicStates.size());
    dynamicStateInfo.pDynamicStates = dynamicStates.data();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{};

    inputAssemblyInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};

    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(vkSwapchainExtent.width);
    viewport.height = static_cast<float>(vkSwapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};

    scissor.offset = {0, 0};
    scissor.extent = vkSwapchainExtent;

    VkPipelineViewportStateCreateInfo viewportStateInfo{};

    viewportStateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportStateInfo.viewportCount = 1;
    viewportStateInfo.pViewports = &viewport;
    viewportStateInfo.scissorCount = 1;
    viewportStateInfo.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizerInfo{};

    rasterizerInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizerInfo.depthClampEnable = VK_FALSE;
    rasterizerInfo.rasterizerDiscardEnable = VK_FALSE;
    rasterizerInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizerInfo.lineWidth = 1.0f;
    rasterizerInfo.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizerInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizerInfo.depthBiasEnable = VK_FALSE;
    rasterizerInfo.depthBiasConstantFactor = 0.0f;
    rasterizerInfo.depthBiasClamp = 0.0f;
    rasterizerInfo.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisamplingInfo{};

    multisamplingInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisamplingInfo.sampleShadingEnable = VK_FALSE;
    multisamplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisamplingInfo.minSampleShading = 1.0f;
    multisamplingInfo.pSampleMask = nullptr;
    multisamplingInfo.alphaToCoverageEnable = VK_FALSE;
    multisamplingInfo.alphaToOneEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};

    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlendingInfo{};

    colorBlendingInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendingInfo.logicOpEnable = VK_FALSE;
    colorBlendingInfo.logicOp = VK_LOGIC_OP_COPY;
    colorBlendingInfo.attachmentCount = 1;
    colorBlendingInfo.pAttachments = &colorBlendAttachment;
    colorBlendingInfo.blendConstants[0] = 0.0f;
    colorBlendingInfo.blendConstants[1] = 0.0f;
    colorBlendingInfo.blendConstants[2] = 0.0f;
    colorBlendingInfo.blendConstants[3] = 0.0f;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};

    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &vkDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(vkDevice, &pipelineLayoutInfo, nullptr,
                               &vkPipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create pipeline layout");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};

    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
    pipelineInfo.pViewportState = &viewportStateInfo;
    pipelineInfo.pRasterizationState = &rasterizerInfo;
    pipelineInfo.pMultisampleState = &multisamplingInfo;
    pipelineInfo.pDepthStencilState = nullptr;
    pipelineInfo.pColorBlendState = &colorBlendingInfo;
    pipelineInfo.pDynamicState = &dynamicStateInfo;
    pipelineInfo.layout = vkPipelineLayout;
    pipelineInfo.renderPass = vkRenderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    if (vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &vkGraphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create graphics pipeline");
    }

    vkDestroyShaderModule(vkDevice, vertexShaderModule, nullptr);
    vkDestroyShaderModule(vkDevice, fragmentShaderModule, nullptr);
  }

  VkShaderModule createVulkanShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};

    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(vkDevice, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create shader module");
    }

    return shaderModule;
  }

  void createVulkanVertexBuffer() {
    VkDeviceSize bufferSize = mesh.vertexBufferSize();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createVulkanBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(vkDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, mesh.vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(vkDevice, stagingBufferMemory);

    createVulkanBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vkVertexBuffer,
        vkVertexBufferMemory);
    copyVulkanBuffer(stagingBuffer, vkVertexBuffer, bufferSize);

    vkDestroyBuffer(vkDevice, stagingBuffer, nullptr);
    vkFreeMemory(vkDevice, stagingBufferMemory, nullptr);
  }

  void createVulkanIndexBuffer() {
    VkDeviceSize bufferSize = mesh.indexBufferSize();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createVulkanBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(vkDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, mesh.indices.data(), (size_t)bufferSize);
    vkUnmapMemory(vkDevice, stagingBufferMemory);

    createVulkanBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vkIndexBuffer,
        vkIndexBufferMemory);
    copyVulkanBuffer(stagingBuffer, vkIndexBuffer, bufferSize);

    vkDestroyBuffer(vkDevice, stagingBuffer, nullptr);
    vkFreeMemory(vkDevice, stagingBufferMemory, nullptr);
  }

  void createVulkanBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags properties, VkBuffer& buffer,
                          VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};

    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.flags = 0;

    if (vkCreateBuffer(vkDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create buffer");
    }

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkDevice, buffer, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{};

    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memoryRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(vkDevice, &allocInfo, nullptr, &bufferMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to allocate buffer memory");
    }

    vkBindBufferMemory(vkDevice, buffer, bufferMemory, 0);
  }

  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties physicalMemoryProperties;
    vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice,
                                        &physicalMemoryProperties);

    for (uint32_t i = 0; i < physicalMemoryProperties.memoryTypeCount; i++) {
      if (typeFilter & (1 << i) &&
          (physicalMemoryProperties.memoryTypes[i].propertyFlags &
           properties) == properties) {
        return i;
      }
    }

    throw std::runtime_error("Failed to find suitable memory type");
  }

  void copyVulkanBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};

    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = vkCommandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(vkDevice, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};

    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};

    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;

    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};

    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    // TODO: use separate queue other than graphics queue for copying buffers.
    vkQueueSubmit(vkGraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(vkGraphicsQueue);

    vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &commandBuffer);
  }

  void createVulkanFramebuffers() {
    vkSwapchainFramebuffers.resize(vkSwapchainImageViews.size());

    for (size_t i = 0; i < vkSwapchainImageViews.size(); i++) {
      VkImageView attachments[] = {vkSwapchainImageViews[i]};

      VkFramebufferCreateInfo framebufferInfo{};

      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferInfo.renderPass = vkRenderPass;
      framebufferInfo.attachmentCount = 1;
      framebufferInfo.pAttachments = attachments;
      framebufferInfo.width = vkSwapchainExtent.width;
      framebufferInfo.height = vkSwapchainExtent.height;
      framebufferInfo.layers = 1;

      if (vkCreateFramebuffer(vkDevice, &framebufferInfo, nullptr,
                              &vkSwapchainFramebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create framebuffer");
      }
    }
  }

  void createVulkanCommandPool() {
    VkPhysicalDeviceQueueFamilies queueFamilies;
    readVulkanPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice,
                                                  queueFamilies);

    VkCommandPoolCreateInfo poolInfo{};

    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilies.graphicsFamily.value();

    if (vkCreateCommandPool(vkDevice, &poolInfo, nullptr, &vkCommandPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create command pool");
    }
  }

  void createVulkanFrameRenderResources() {
    vkFrameRenderResources.resize(MAX_FRAMES_IN_FLIGHT);

    for (auto& resource : vkFrameRenderResources) {
      createVulkanCommandBuffer(resource.commandBuffer);
      createVulkanSemaphore(resource.imageAvailableSemaphore);
      createVulkanSemaphore(resource.renderFinishedSemaphore);
      createVulkanFence(resource.inFlightFence);
      createVulkanBuffer(sizeof(UniformBufferObject),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         resource.uniformBuffer, resource.uniformBefferMemory);
      vkMapMemory(vkDevice, resource.uniformBefferMemory, 0,
                  sizeof(UniformBufferObject), 0,
                  &resource.uniformBufferMapped);
    }
  }

  void createVulkanCommandBuffer(VkCommandBuffer& commandBuffer) {
    VkCommandBufferAllocateInfo allocInfo{};

    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = vkCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(vkDevice, &allocInfo, &commandBuffer) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to allocate command buffer");
    }
  }

  void createVulkanSemaphore(VkSemaphore& semaphore) {
    VkSemaphoreCreateInfo semaphoreInfo{};

    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    if (vkCreateSemaphore(vkDevice, &semaphoreInfo, nullptr, &semaphore) !=
        VK_SUCCESS) {
      throw std::runtime_error("Failed to create semaphore");
    }
  }

  void createVulkanFence(VkFence& fence) {
    VkFenceCreateInfo fenceInfo{};

    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags =
        VK_FENCE_CREATE_SIGNALED_BIT;  // Start in signaled state, so we don't
                                       // wait on first frame

    if (vkCreateFence(vkDevice, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create fence");
    }
  }

  void createVulkanDescriptorPool() {
    VkDescriptorPoolSize poolSize{};

    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    VkDescriptorPoolCreateInfo poolInfo{};

    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    if (vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr,
                               &vkDescriptorPool) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create descriptor pool");
    }
  }

  void createVulkanDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                               vkDescriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};

    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = vkDescriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    allocInfo.pSetLayouts = layouts.data();

    vkDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(vkDevice, &allocInfo,
                                 vkDescriptorSets.data()) != VK_SUCCESS) {
      throw std::runtime_error("Failed to create descriptor sets");
    }
  }

  void recordVulkanCommandBuffer(VkCommandBuffer commandBuffer,
                                 uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};

    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;
    beginInfo.pInheritanceInfo = nullptr;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
      throw std::runtime_error("Failed to begin recording command buffer");
    }

    VkRenderPassBeginInfo renderPassInfo{};

    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = vkRenderPass;
    renderPassInfo.framebuffer = vkSwapchainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = vkSwapchainExtent;
    VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      vkGraphicsPipeline);

    VkBuffer vertexBuffers[] = {vkVertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, vkIndexBuffer, 0, VK_INDEX_TYPE_UINT16);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(vkSwapchainExtent.width);
    viewport.height = static_cast<float>(vkSwapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = vkSwapchainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(mesh.numIndices()), 1,
                     0, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
      throw std::runtime_error("Failed to record command buffer");
    }
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(glfwWindow)) {
      glfwPollEvents();
      frame();
    }

    vkDeviceWaitIdle(vkDevice);
  }

  void frame() {
    uint32_t frameIndex = currentFrame % MAX_FRAMES_IN_FLIGHT;
    currentFrame++;

    VkFrameRenderResources frameResources = vkFrameRenderResources[frameIndex];

    // Wait for previous frame to finish
    vkWaitForFences(vkDevice, 1, &frameResources.inFlightFence, VK_TRUE,
                    UINT64_MAX);

    // Acquire image from swapchain
    uint32_t imageIndex;
    VkResult acquireResult = vkAcquireNextImageKHR(
        vkDevice, vkSwapchain, UINT64_MAX,
        frameResources.imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
      recreateVulkanSwapchain();
      return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
      throw std::runtime_error("Failed to acquire swapchain image");
    }

    // Update uniform buffer
    updateUniformBuffer(frameIndex);

    // Submit command buffer
    vkResetCommandBuffer(frameResources.commandBuffer, 0);
    recordVulkanCommandBuffer(frameResources.commandBuffer, imageIndex);

    VkSubmitInfo submitInfo{};

    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {frameResources.imageAvailableSemaphore};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &frameResources.commandBuffer;
    VkSemaphore signalSemaphores[] = {frameResources.renderFinishedSemaphore};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vkResetFences(vkDevice, 1, &frameResources.inFlightFence);
    if (vkQueueSubmit(vkGraphicsQueue, 1, &submitInfo,
                      frameResources.inFlightFence) != VK_SUCCESS) {
      throw std::runtime_error("Failed to submit draw command buffer");
    }

    // Present image to swapchain
    VkPresentInfoKHR presentInfo{};

    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapchains[] = {vkSwapchain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;

    VkResult presentResult = vkQueuePresentKHR(vkPresentQueue, &presentInfo);

    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
        presentResult == VK_SUBOPTIMAL_KHR) {
      recreateVulkanSwapchain();
    } else if (presentResult != VK_SUCCESS) {
      throw std::runtime_error("Failed to present swapchain image");
    }
  }

  void updateUniformBuffer(uint32_t currentImageIndex) {
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     currentTime - startTime)
                     .count();

    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view =
        glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(
        glm::radians(45.0f),
        vkSwapchainExtent.width / (float)vkSwapchainExtent.height, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    memcpy(vkFrameRenderResources[currentImageIndex].uniformBufferMapped, &ubo,
           sizeof(ubo));
  }

  void recreateVulkanSwapchain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(glfwWindow, &width, &height);

    // Wait until window is back to foreground
    while (width == 0 || height == 0) {
      spdlog::info("Window minimized, waiting.");
      glfwGetFramebufferSize(glfwWindow, &width, &height);
      glfwWaitEvents();
    }

    vkDeviceWaitIdle(vkDevice);

    cleanupVulkanSwapchain();

    spdlog::info("Recreating swapchain with new dimensions {}x{}.", width,
                 height);

    createVulkanSwapchain();
    createVulkanImageViews();
    createVulkanFramebuffers();
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
    cleanupVulkanSwapchain();

    vkDestroyBuffer(vkDevice, vkVertexBuffer, nullptr);
    vkFreeMemory(vkDevice, vkVertexBufferMemory, nullptr);
    vkDestroyBuffer(vkDevice, vkIndexBuffer, nullptr);
    vkFreeMemory(vkDevice, vkIndexBufferMemory, nullptr);
    for (const auto& resource : vkFrameRenderResources) {
      vkDestroySemaphore(vkDevice, resource.imageAvailableSemaphore, nullptr);
      vkDestroySemaphore(vkDevice, resource.renderFinishedSemaphore, nullptr);
      vkDestroyFence(vkDevice, resource.inFlightFence, nullptr);
      vkDestroyBuffer(vkDevice, resource.uniformBuffer, nullptr);
      vkFreeMemory(vkDevice, resource.uniformBefferMemory, nullptr);
    }
    vkDestroyDescriptorPool(vkDevice, vkDescriptorPool, nullptr);
    vkDestroyCommandPool(vkDevice, vkCommandPool, nullptr);
    vkDestroyPipeline(vkDevice, vkGraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(vkDevice, vkPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(vkDevice, vkDescriptorSetLayout, nullptr);
    vkDestroyRenderPass(vkDevice, vkRenderPass, nullptr);
    vkDestroyDevice(vkDevice, nullptr);
    vkDestroySurfaceKHR(vkInstance, vkSurface, nullptr);

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

  void cleanupVulkanSwapchain() {
    for (auto framebuffer : vkSwapchainFramebuffers) {
      vkDestroyFramebuffer(vkDevice, framebuffer, nullptr);
    }

    for (auto imageView : vkSwapchainImageViews) {
      vkDestroyImageView(vkDevice, imageView, nullptr);
    }

    vkDestroySwapchainKHR(vkDevice, vkSwapchain, nullptr);
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

  static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
      throw std::runtime_error(
          std::format("Failed to open file: {}", filename));
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
  }
};

int main() {
  Application app;

  spdlog::set_level(spdlog::level::info);

  try {
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
