#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <spdlog/spdlog.h>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstdlib>

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

	void initVulkan() {
		createVulkanInstance();
	}

	void createVulkanInstance() {
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		uint32_t numGlfwExtensions = 0;
		const char** glfwExtensions = nullptr;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&numGlfwExtensions);

		createInfo.enabledExtensionCount = numGlfwExtensions;
		createInfo.ppEnabledExtensionNames = glfwExtensions;
		createInfo.enabledLayerCount = 0;

		uint32_t numVulkanExtensions = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &numVulkanExtensions, nullptr);
		std::vector<VkExtensionProperties> vulkanExtensions(numVulkanExtensions);
		vkEnumerateInstanceExtensionProperties(nullptr, &numVulkanExtensions, vulkanExtensions.data());

		spdlog::debug("Available extensions:");
		for (const auto& extension : vulkanExtensions) {
			spdlog::debug("\t{}", extension.extensionName);
		}

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
				throw new std::runtime_error(std::format("Extension {} required by GLFW is not supported by Vulkan", glfwExtension));
			}
		}

		if (vkCreateInstance(&createInfo, nullptr, &vkInstance) != VK_SUCCESS) {
			throw new std::runtime_error("Failed to create Vulkan instance");
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
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
