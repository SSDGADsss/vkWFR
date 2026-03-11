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

  struct aBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
  };

  aBuffer calbuffer;

  uint64_t inputSize, outputSize, bufferSize;

public:
  FFT_R2C_2D(int width, int height, const VkInstance &instance,
             const VkPhysicalDevice &phydevice, const VkDevice &device,
             uint32_t computeQueueFamilyIndex);

  void operator()(std::shared_ptr<kp::TensorT<double>> input,
                  std::shared_ptr<kp::TensorT<double>> output);

  ~FFT_R2C_2D();
};

#endif
