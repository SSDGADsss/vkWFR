#ifndef FFT_R2C_2D_H
#define FFT_R2C_2D_H

#include "kompute/Tensor.hpp"
#include "vulkan/vulkan_core.h"
#include <memory>
#include <vkFFT.h>

class FFT_R2C_2D {
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

  VkCommandBuffer commandBuffer = {};

  struct aBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
  };

  aBuffer calbuffer;

  uint64_t inputSize, outputSize, bufferSize;
  std::shared_ptr<kp::TensorT<float>> input, output;

public:
  FFT_R2C_2D(int width, int height, std::shared_ptr<kp::TensorT<float>> input,
             std::shared_ptr<kp::TensorT<float>> output,
             const VkInstance &instance, const VkPhysicalDevice &phydevice,
             const VkDevice &device, uint32_t computeQueueFamilyIndex);

  void operator()();

  ~FFT_R2C_2D();
};

#endif
