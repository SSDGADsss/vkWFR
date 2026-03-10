#include "vulkan/vulkan_core.h"
#include <FFT_C2C_2D.hpp>
#include <stdexcept>
#include <vkFFT/vkFFT_Structs/vkFFT_Structs.h>
#include <vulkan/vulkan.h>

FFT_C2C_2D::FFT_C2C_2D(int width_, int height_, const VkInstance &instance_,
                       const VkPhysicalDevice &phydevice_,
                       const VkDevice &device_,
                       uint32_t computeQueueFamilyIndex)
    : instance(instance_), phydevice(phydevice_), device(device_),
      width(width_), height(height_), bufferSize(16 * width_ * height_) {
  memset(&configuration, 0, sizeof(configuration));
  memset(&app, 0, sizeof(app));

  vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &queue);

  VkFenceCreateInfo fenceCreateInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fenceCreateInfo.flags = 0;
  if (vkCreateFence(device, &fenceCreateInfo, nullptr, &fence) != VK_SUCCESS)
    throw std::runtime_error("FFT_C2C_2D::constructor vkCreateFence failed");

  VkCommandPoolCreateInfo info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  info.queueFamilyIndex = computeQueueFamilyIndex;
  if (vkCreateCommandPool(device, &info, nullptr, &commandPool) != VK_SUCCESS)
    throw std::runtime_error(
        "FFT_C2C_2D::constructor vkCreateCommandPool failed");

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
  configuration.doublePrecision = 1;
  configuration.normalize = 1; // 归一化

  configuration.bufferSize = &bufferSize;
  configuration.bufferNum = 1;

  if (initializeVkFFT(&app, configuration) != VKFFT_SUCCESS)
    throw std::runtime_error("FFT_C2C_2D::constructor initializeVkFFT failed");
}

void FFT_C2C_2D::forward(std::shared_ptr<kp::TensorT<double>> data) {
  VkFFTLaunchParams launchParams = {};
  launchParams.buffer = (VkBuffer *)data->getPrimaryBuffer().get();

  {
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = {};
    if (vkAllocateCommandBuffers(device, &commandBufferAllocateInfo,
                                 &commandBuffer) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::forward vkAllocateCommandBuffers call is failed");

    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) !=
        VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::forward vkBeginCommandBuffer is failed");
    launchParams.commandBuffer = &commandBuffer;

    // NOTE: 这里可以添加多个FFT步骤用于优化,但要更改批次数量
    if (VkFFTAppend(&app, -1, &launchParams) != VKFFT_SUCCESS)
      throw std::runtime_error("FFT_C2C_2D::forward VkFFTAppend is failed");
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::forward vkEndCommandBuffer is failed");

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::forward vkQueueSubmit call is failed");
    if (vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::forward vkWaitForFences call is failed");
    if (vkResetFences(device, 1, &fence) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::forward vkResetFences call is failed");
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  }
}

void FFT_C2C_2D::inverse(std::shared_ptr<kp::TensorT<double>> data) {
  VkFFTLaunchParams launchParams = {};
  launchParams.buffer = (VkBuffer *)data->getPrimaryBuffer().get();

  {
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = {};
    if (vkAllocateCommandBuffers(device, &commandBufferAllocateInfo,
                                 &commandBuffer) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::inverse vkAllocateCommandBuffers call is failed");

    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) !=
        VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::inverse vkBeginCommandBuffer is failed");
    launchParams.commandBuffer = &commandBuffer;

    // NOTE: 这里可以添加多个FFT步骤用于优化,但要更改批次数量
    if (VkFFTAppend(&app, 1, &launchParams) != VKFFT_SUCCESS)
      throw std::runtime_error("FFT_C2C_2D::inverse VkFFTAppend is failed");
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::inverse vkEndCommandBuffer is failed");

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    if (vkQueueSubmit(queue, 1, &submitInfo, fence) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::inverse vkQueueSubmit call is failed");
    if (vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::inverse vkWaitForFences call is failed");
    if (vkResetFences(device, 1, &fence) != VK_SUCCESS)
      throw std::runtime_error(
          "FFT_C2C_2D::inverse vkResetFences call is failed");
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  }
}

FFT_C2C_2D::~FFT_C2C_2D() {
  // 清理vkFFT应用
  deleteVkFFT(&app);

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
