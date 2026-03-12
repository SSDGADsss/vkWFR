#ifndef FFT_C2C_2D_H
#define FFT_C2C_2D_H

#include "kompute/Tensor.hpp"
#include "vulkan/vulkan_core.h"
#include <memory>
#include <vkFFT.h>

class FFT_C2C_2D {
  VkFFTApplication app = {};
  VkFFTLaunchParams launchParams = {};
  VkFFTConfiguration configuration = {};
  int width, height;

  VkInstance instance;
  VkPhysicalDevice phydevice;
  VkDevice device;
  VkQueue queue;
  VkFence fence;
  VkCommandPool commandPool;

  uint64_t bufferSize;
  std::shared_ptr<kp::TensorT<float>> data;

  VkCommandBuffer commandBuffer_forward = {};
  VkCommandBuffer commandBuffer_inverse = {};

public:
  FFT_C2C_2D(int width, int height, std::shared_ptr<kp::TensorT<float>> data,
             const VkInstance &instance, const VkPhysicalDevice &phydevice,
             const VkDevice &device, uint32_t computeQueueFamilyIndex);

  // WARN: forward和inverse不要并发调用
  void forward();
  // WARN: forward和inverse不要并发调用
  void inverse();

  void asyncInverse();
  void asyncForward();
  void waitAsync();

  ~FFT_C2C_2D();
};

#endif
