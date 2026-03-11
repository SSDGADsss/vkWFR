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

public:
  FFT_C2C_2D(int width, int height, const VkInstance &instance,
             const VkPhysicalDevice &phydevice, const VkDevice &device,
             uint32_t computeQueueFamilyIndex);

  void forward(std::shared_ptr<kp::TensorT<double>> data);
  void inverse(std::shared_ptr<kp::TensorT<double>> data);

  ~FFT_C2C_2D();
};

#endif
