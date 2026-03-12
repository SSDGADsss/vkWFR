#include "vulkan/vulkan_core.h"
#include <FFT_R2C_2D.hpp>
#include <stdexcept>
#include <vkFFT/vkFFT_Structs/vkFFT_Structs.h>
#include <vulkan/vulkan.h>

// 创建 GPU 存储缓冲区
static VkBuffer createStorageBuffer(VkDevice device,
                                    VkPhysicalDevice physicalDevice,
                                    VkDeviceSize size, VkDeviceMemory *memory) {

  // 1. 创建缓冲区
  VkBufferCreateInfo bufferInfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage =
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  VkBuffer buffer;
  VkResult result = vkCreateBuffer(device, &bufferInfo, NULL, &buffer);
  if (result != VK_SUCCESS) {
    fprintf(stderr, "Failed to create buffer\n");
    return VK_NULL_HANDLE;
  }

  // 2. 获取内存需求
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

  // 3. 查找设备本地内存类型
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  uint32_t memoryTypeIndex = -1;
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((memRequirements.memoryTypeBits & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags &
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      memoryTypeIndex = i;
      break;
    }
  }

  if (memoryTypeIndex == -1) {
    fprintf(stderr, "Failed to find suitable memory type\n");
    vkDestroyBuffer(device, buffer, NULL);
    return VK_NULL_HANDLE;
  }

  // 4. 分配内存
  VkMemoryAllocateInfo allocInfo = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memRequirements.size,
      .memoryTypeIndex = memoryTypeIndex,
  };

  result = vkAllocateMemory(device, &allocInfo, NULL, memory);
  if (result != VK_SUCCESS) {
    fprintf(stderr, "Failed to allocate memory\n");
    vkDestroyBuffer(device, buffer, NULL);
    return VK_NULL_HANDLE;
  }

  // 5. 绑定内存
  result = vkBindBufferMemory(device, buffer, *memory, 0);
  if (result != VK_SUCCESS) {
    fprintf(stderr, "Failed to bind buffer memory\n");
    vkFreeMemory(device, *memory, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    return VK_NULL_HANDLE;
  }

  return buffer;
}

FFT_R2C_2D::FFT_R2C_2D(int width_, int height_,
                       std::shared_ptr<kp::TensorT<float>> input_,
                       std::shared_ptr<kp::TensorT<float>> output_,
                       const VkInstance &instance_,
                       const VkPhysicalDevice &phydevice_,
                       const VkDevice &device_,
                       uint32_t computeQueueFamilyIndex)
    : instance(instance_), phydevice(phydevice_), device(device_),
      width(width_), height(height_),
      inputSize(sizeof(float) * width_ * height_),
      outputSize(sizeof(float) * width_ * height_ * 2),
      bufferSize(sizeof(float) * 2 * (width_ / 2 + 1) * height_), input(input_),
      output(output_) {
  memset(&configuration, 0, sizeof(configuration));
  memset(&app, 0, sizeof(app));

  vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &queue);

  VkFenceCreateInfo fenceCreateInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fenceCreateInfo.flags = 0;
  if (vkCreateFence(device, &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS)
    throw std::runtime_error("FFT_R2C_2D::constructor vkCreateFence failed");

  VkCommandPoolCreateInfo info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  info.queueFamilyIndex = computeQueueFamilyIndex;
  if (vkCreateCommandPool(device, &info, nullptr, &commandPool) != VK_SUCCESS)
    throw std::runtime_error(
        "FFT_R2C_2D::constructor vkCreateCommandPool failed");

  configuration.queue = &queue;
  configuration.fence = &fence;
  configuration.device = &device;
  configuration.commandPool = &commandPool;
  configuration.physicalDevice = &phydevice;
  configuration.isCompilerInitialized = false;
  configuration.FFTdim = 2;
  configuration.size[0] = width;
  configuration.size[1] = height;
  configuration.numberBatches = 1;
  configuration.performR2C = 1;
  configuration.performDCT = 0;
  configuration.doublePrecision = 0;

  configuration.isInputFormatted = 1;
  configuration.inputBufferNum = 1;
  configuration.inputBufferSize = &inputSize;
  configuration.inputBufferStride[0] = configuration.size[0];
  configuration.inputBufferStride[1] =
      configuration.size[0] * configuration.size[1];

  configuration.isOutputFormatted = 1;
  configuration.outputBufferNum = 1;
  configuration.outputBufferSize = &outputSize;
  configuration.outputBufferStride[0] = configuration.size[0];
  configuration.outputBufferStride[1] =
      configuration.size[0] * configuration.size[1];

  configuration.bufferSize = &bufferSize;
  configuration.bufferNum = 1;

  if (initializeVkFFT(&app, configuration) != VKFFT_SUCCESS)
    throw std::runtime_error("FFT_R2C_2D::constructor initializeVkFFT failed");

  calbuffer.buffer =
      createStorageBuffer(device, phydevice, bufferSize, &calbuffer.memory);
  if (calbuffer.buffer == VK_NULL_HANDLE)
    throw std::runtime_error(
        "FFT_R2C_2D::constructor createStorageBuffer failed");

  VkFFTLaunchParams launchParams = {};
  launchParams.buffer = &calbuffer.buffer;
  launchParams.inputBuffer = (VkBuffer *)input->getPrimaryBuffer().get();
  launchParams.outputBuffer = (VkBuffer *)output->getPrimaryBuffer().get();

  {
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &commandBufferAllocateInfo,
                                 &commandBuffer) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_R2C_2D::operator vkAllocateCommandBuffers call is failed");

    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) !=
        VK_SUCCESS)
      throw std::runtime_error(
          "FFT_R2C_2D::operator vkBeginCommandBuffer is failed");
    launchParams.commandBuffer = &commandBuffer;

    if (VkFFTAppend(&app, -1, &launchParams) != VKFFT_SUCCESS)
      throw std::runtime_error("FFT_R2C_2D::operator VkFFTAppend is failed");
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_R2C_2D::operator vkEndCommandBuffer is failed");
  }
}

void FFT_R2C_2D::operator()() {
  {
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_R2C_2D::operator vkQueueSubmit call is failed");
    if (vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_R2C_2D::operator vkWaitForFences call is failed");
    if (vkResetFences(device, 1, &fence) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_R2C_2D::operator vkResetFences call is failed");
  }
}

FFT_R2C_2D::~FFT_R2C_2D() {

  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  // 清理vkFFT应用
  deleteVkFFT(&app);

  // 销毁缓冲区
  if (calbuffer.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device, calbuffer.buffer, nullptr);
    calbuffer.buffer = VK_NULL_HANDLE;
  }

  // 释放内存
  if (calbuffer.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device, calbuffer.memory, nullptr);
    calbuffer.memory = VK_NULL_HANDLE;
  }

  // 销毁命令池
  if (commandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device, commandPool, nullptr);
    commandPool = VK_NULL_HANDLE;
  }

  // 销毁fence
  if (fence != VK_NULL_HANDLE) {
    vkDestroyFence(device, fence, nullptr);
    fence = VK_NULL_HANDLE;
  }

  // 注意：instance, phydevice, device, queue 是外部传入的，不应在此销毁
}
