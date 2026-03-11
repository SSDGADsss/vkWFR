#ifndef VKWFR_H
#define VKWFR_H

#include <FFT_C2C_2D.hpp>
#include <FFT_R2C_2D.hpp>
#include <array>
#include <kompute/Kompute.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

class vkWFR {
  const int imgWidth, imgHeight;
  const std::array<int, 4> ROI;

  VkInstance raw_instance;
  VkPhysicalDevice raw_phydevice;
  VkDevice raw_device;
  VkQueue raw_compute_queue;
  uint32_t computeQueueFamilyIndex = UINT32_MAX;

  std::unique_ptr<kp::Manager> mgr;

  std::shared_ptr<vk::Instance> instance;
  std::shared_ptr<vk::PhysicalDevice> physicalDevice;
  std::shared_ptr<vk::Device> device;
  std::shared_ptr<vk::Queue> compute_queue;

  const unsigned int mm, nn;
  const int sx, sy;
  const int cal_width, cal_height;
  const int sigmax, sigmay;
  const double wxl, wxi, wxh, wyl, wyi, wyh, thr;

  std::shared_ptr<kp::TensorT<double>> GaussianWindow;
  std::shared_ptr<kp::TensorT<double>> FfBuffer;
  std::shared_ptr<kp::TensorT<double>> w_expanded;
  std::shared_ptr<kp::TensorT<double>> result_ridge;
  std::shared_ptr<kp::TensorT<double>> calReady;
  std::shared_ptr<kp::TensorT<unsigned char>> tensorIn;

  std::shared_ptr<kp::Algorithm> algo_cal_w;
  std::shared_ptr<kp::Algorithm> algo_cal_mulmatrix;
  std::shared_ptr<kp::Algorithm> algo_cal_ridge;
  std::shared_ptr<kp::Algorithm> algo_cal_hermitian;
  std::shared_ptr<kp::Algorithm> algo_cal_init;

  std::shared_ptr<kp::Sequence> recorder_init;
  std::shared_ptr<kp::Sequence> recorder_hermitian;
  std::shared_ptr<kp::Sequence> recorder_mulmatrix;
  std::shared_ptr<kp::Sequence> recorder_cal_ridge;
  std::shared_ptr<kp::Sequence> recorder_base_func;

  std::shared_ptr<kp::OpBase> memreset;
  std::shared_ptr<kp::OpBase> initMem;

  std::unique_ptr<FFT_R2C_2D> fft_r2c;
  std::unique_ptr<FFT_C2C_2D> fft_c2c;

public:
  vkWFR(int imgwidth, int imgheight, std::array<int, 4> ROI, int sigmax,
        double wxl, double wxi, double wxh, int sigmay, double wyl, double wyi,
        double wyh, double thr);
  ~vkWFR();

  std::vector<double> operator()(std::vector<unsigned char> image);
};

#endif
