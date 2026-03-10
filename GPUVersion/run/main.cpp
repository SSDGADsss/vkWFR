#include "kompute/Sequence.hpp"
#include "kompute/Tensor.hpp"
#include "kompute/operations/OpAlgoDispatch.hpp"
#include "kompute/operations/OpSyncDevice.hpp"
#include "kompute/operations/OpSyncLocal.hpp"
#include "vulkan/vulkan_core.h"
#include "vulkan/vulkan_handles.hpp"
#include <Eigen/Eigen>
#include <Eigen/src/Core/util/Constants.h>
#include <chrono>
#include <complex>
#include <cstdint>
#include <fftw3.h>
#include <fstream>
#include <iostream>
#include <kompute/Kompute.hpp>
#include <memory>
#include <shader/step1.hpp>
#include <shader/step2.hpp>
#include <shader/step3.hpp>
#include <shader/step4.hpp>
#include <stdexcept>
#include <vkFFT.h>
#include <vkFFT/vkFFT_AppManagement/vkFFT_RunApp.h>
#include <vkFFT/vkFFT_Structs/vkFFT_Structs.h>
#include <vulkan/vulkan.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

// 类型转换+扩展+ROI
const static std::vector<uint32_t> shader_step1(comp::STEP1_COMP_SPV.begin(),
                                                comp::STEP1_COMP_SPV.end());
// Hermitian对称性处理
const static std::vector<uint32_t> shader_step2(comp::STEP2_COMP_SPV.begin(),
                                                comp::STEP2_COMP_SPV.end());
// 创建高斯窗
const static std::vector<uint32_t> shader_step3(comp::STEP3_COMP_SPV.begin(),
                                                comp::STEP3_COMP_SPV.end());
// 高斯窗归一化
const static std::vector<uint32_t> shader_step4(comp::STEP4_COMP_SPV.begin(),
                                                comp::STEP4_COMP_SPV.end());
constexpr unsigned int HandleBlockSize = 1024;

// 定义复数类型
typedef std::complex<double> Complex;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> MatrixXcd;

struct wfrResult {
  Eigen::MatrixXd wx, wy, phase, r;
};

// 辅助函数：扩展矩阵到指定尺寸（类似MATLAB的fexpand函数）
MatrixXcd fexpand(const MatrixXcd &f, int mm, int nn) {
  // 获取原始矩阵尺寸
  int m = f.rows();
  int n = f.cols();

  // 创建扩展矩阵并初始化为0
  MatrixXcd expanded = MatrixXcd::Zero(mm, nn);

  // 将原始数据复制到扩展矩阵的左上角
  expanded.block(0, 0, m, n) = f;

  return expanded;
}

// 辅助函数：计算矩阵的2D FFT
MatrixXcd fft2(const MatrixXcd &input) {
  int rows = input.rows();
  int cols = input.cols();

  // 分配FFTW输入输出数组
  fftw_complex *in =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * rows * cols);
  fftw_complex *out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * rows * cols);

  // 复制数据到输入数组
  for (int i = 0; i < rows * cols; ++i) {
    in[i][0] = input(i).real();
    in[i][1] = input(i).imag();
  }

  // 创建FFT计划
  fftw_plan plan =
      fftw_plan_dft_2d(cols, rows, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

  // 执行FFT
  fftw_execute(plan);

  // 将结果复制回Eigen矩阵
  MatrixXcd result(rows, cols);
  for (int i = 0; i < rows * cols; ++i) {
    result(i) = Complex(out[i][0], out[i][1]);
  }

  // 清理
  fftw_destroy_plan(plan);
  fftw_free(in);
  fftw_free(out);

  return result;
}

// 辅助函数：计算矩阵的2D逆FFT
MatrixXcd ifft2(const MatrixXcd &input) {
  int rows = input.rows();
  int cols = input.cols();

  // 分配FFTW输入输出数组
  fftw_complex *in =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * rows * cols);
  fftw_complex *out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * rows * cols);

  // 复制数据到输入数组
  for (int i = 0; i < rows * cols; ++i) {
    in[i][0] = input(i).real();
    in[i][1] = input(i).imag();
  }

  // 创建逆FFT计划
  fftw_plan plan =
      fftw_plan_dft_2d(cols, rows, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

  // 执行逆FFT
  fftw_execute(plan);

  // 将结果复制回Eigen矩阵并归一化
  MatrixXcd result(rows, cols);
  double norm = 1.0 / (rows * cols);
  for (int i = 0; i < rows * cols; ++i)
    result(i) = Complex(out[i][0] * norm, out[i][1] * norm);

  // 清理
  fftw_destroy_plan(plan);
  fftw_free(in);
  fftw_free(out);

  return result;
}

// 主函数：onlyWFR实现
wfrResult onlyWFR(const std::span<uint8_t> f, int width, int height,
                  std::array<unsigned int, 4> ROI, int sigmax, double wxl,
                  double wxi, double wxh, int sigmay, double wyl, double wyi,
                  double wyh, double thr) {
  // 注释：onlyWFR函数 - 窗口傅里叶脊提取算法
  // 该函数用于从单个条纹图案中提取相位信息

  wfrResult result;

  // 步骤1：计算窗口半尺寸（类似MATLAB中的round(3*sigmax)）
  int sx = static_cast<int>(std::round(3 * sigmax));
  int sy = static_cast<int>(std::round(3 * sigmay));

  // WARN: 这里改变过语义
  const unsigned int mm = ROI[2] + 2 * sx;
  const unsigned int nn = ROI[3] + 2 * sy;

  std::vector<unsigned int> intImage(f.size());
  for (int i = 0; i < f.size(); i++) {
    intImage[i] = f[i];
  }

  VkInstance raw_instance;
  VkPhysicalDevice raw_phydevice;
  VkDevice raw_device;
  VkQueue raw_compute_queue;
  uint32_t computeQueueFamilyIndex = UINT32_MAX;
  {
    // 初始化Vulkan实例
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "vkWFR";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    // 启用必要的扩展
    std::vector<const char *> instanceExtensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};

    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;
    instanceCreateInfo.enabledExtensionCount =
        static_cast<uint32_t>(instanceExtensions.size());
    instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();

    // 创建实例
    VkResult result =
        vkCreateInstance(&instanceCreateInfo, nullptr, &raw_instance);
    if (result != VK_SUCCESS) {
      std::cerr << "Failed to create Vulkan instance: " << result << std::endl;
      throw std::runtime_error("Failed to create Vulkan instance");
    }

    // 枚举物理设备
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(raw_instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
      std::cerr << "No Vulkan physical devices found" << std::endl;
      throw std::runtime_error("No Vulkan physical devices found");
    }

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(raw_instance, &deviceCount,
                               physicalDevices.data());

    // 选择第0个设备
    raw_phydevice = physicalDevices[0];

    // 准备设备扩展
    std::vector<const char *> deviceExtensions = {
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME};

    // 获取队列族属性
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(raw_phydevice, &queueFamilyCount,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(raw_phydevice, &queueFamilyCount,
                                             queueFamilies.data());

    // 查找支持计算队列的队列族
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        computeQueueFamilyIndex = i;
        break;
      }
    }

    if (computeQueueFamilyIndex == UINT32_MAX) {
      std::cerr << "No compute queue family found" << std::endl;
      throw std::runtime_error("No compute queue family found");
    }

    // 这里强制支持32位缓冲区浮点加法运算
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures = {
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .shaderSharedFloat32Atomics = VK_TRUE,
        .shaderSharedFloat32AtomicAdd = VK_TRUE};

    VkPhysicalDeviceFeatures2 deviceFeatures2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &atomicFloatFeatures};

    // 创建逻辑设备
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.pNext = &deviceFeatures2;
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount =
        static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    result =
        vkCreateDevice(raw_phydevice, &deviceCreateInfo, nullptr, &raw_device);
    if (result != VK_SUCCESS)
      throw std::runtime_error("Failed to create logical device");

    // 获取计算队列句柄
    vkGetDeviceQueue(raw_device, computeQueueFamilyIndex, 0,
                     &raw_compute_queue);

    std::cout << "Vulkan initialized successfully" << std::endl;
  }

  auto instance = std::make_shared<vk::Instance>(raw_instance);
  auto physicalDevice = std::make_shared<vk::PhysicalDevice>(raw_phydevice);
  auto device = std::make_shared<vk::Device>(raw_device);
  auto compute_queue = std::make_shared<vk::Queue>(raw_compute_queue);

  // 初始化kompute
  // kp::Manager mgr(0, {}, {"VK_KHR_shader_non_semantic_info"});
  kp::Manager mgr(instance, physicalDevice, device);
  std::shared_ptr<kp::TensorT<unsigned int>> tensorIn =
      mgr.tensorT<unsigned int>(std::move(intImage));

  std::shared_ptr<kp::TensorT<double>> calReady =
      mgr.tensorT<double>(std::vector<double>(mm * nn, 0));

  std::shared_ptr<kp::Algorithm> algo =
      mgr.algorithm<unsigned int, unsigned int>(
          {tensorIn, calReady}, shader_step1, kp::Workgroup({1}),
          {(unsigned int)width, (unsigned int)height, HandleBlockSize, ROI[0],
           ROI[1], ROI[2], ROI[3], (unsigned)mm, (unsigned)nn},
          {0});

  {
    std::shared_ptr<kp::Sequence> recorder(
        new kp::Sequence(physicalDevice, device, compute_queue, 0));
    // auto recorder = mgr.sequence();
    recorder->record<kp::OpSyncDevice>({tensorIn, calReady});
    for (unsigned int i = 0; i < width * height / HandleBlockSize; i++)
      recorder->record<kp::OpAlgoDispatch>(algo, std::vector<unsigned int>{i});
    // 此时calReady就是vkfft计算就绪了
    recorder->eval();
  }

  // NOTE: 在这里初始化vkFFT

  VkFence fence;
  {
    VkFenceCreateInfo fenceCreateInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceCreateInfo.flags = 0;
    if (vkCreateFence(raw_device, &fenceCreateInfo, nullptr, &fence) !=
        VK_SUCCESS)
      throw std::runtime_error("onlyWFR vkCreateFence failed");
  }

  VkCommandPool commandPool;
  {
    VkCommandPoolCreateInfo info = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = computeQueueFamilyIndex;
    if (vkCreateCommandPool(raw_device, &info, nullptr, &commandPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("onlyWFR vkCreateCommandPool failed");
    }
  }

  // NOTE: 计算高斯窗
  const int cal_width = 2 * sx + 1;
  const int cal_height = 2 * sy + 1;
  std::shared_ptr<kp::TensorT<double>> GaussianWindow =
      mgr.tensorT<double>(std::vector<double>(cal_width * cal_height + 1, 0));
  {
    std::shared_ptr<kp::Algorithm> algo1 = mgr.algorithm<int, int>(
        {GaussianWindow}, shader_step3, kp::Workgroup({1}),
        {HandleBlockSize, cal_width, cal_height, sigmax, sigmay}, {0});
    std::shared_ptr<kp::Sequence> recorder(
        new kp::Sequence(physicalDevice, device, compute_queue, 0));
    recorder->record<kp::OpSyncDevice>({GaussianWindow});
    for (int i = 0;
         i < (cal_width * cal_height + HandleBlockSize - 1) / HandleBlockSize;
         i++)
      recorder->record<kp::OpAlgoDispatch>(algo1, std::vector<int>{i});
    recorder->eval();
    // NOTE: 归一化
    std::shared_ptr<kp::Algorithm> algo2 = mgr.algorithm<int, int>(
        {GaussianWindow}, shader_step4, kp::Workgroup({1}),
        {HandleBlockSize, cal_width * cal_height}, {0});
    for (int i = 0;
         i < (cal_width * cal_height + HandleBlockSize - 1) / HandleBlockSize;
         i++)
      recorder->record<kp::OpAlgoDispatch>(algo2, std::vector<int>{i});
    recorder->eval();
    ;
  }

  // 计算并将缓冲区发送到GPU
  uint64_t inputBufferSize = (uint64_t)sizeof(double) * mm * nn;
  uint64_t outputBufferSize = (uint64_t)sizeof(double) * mm * nn * 2;
  uint64_t bufferSize = sizeof(double) * 2 * (mm / 2 + 1) * nn;

  std::shared_ptr<kp::TensorT<double>> FwBuffer =
      mgr.tensorT<double>(std::vector<double>(outputBufferSize));
  std::shared_ptr<kp::TensorT<double>> fftBuffer =
      mgr.tensorT<double>(std::vector<double>(bufferSize));
  {
    std::shared_ptr<kp::Sequence> recorder(
        new kp::Sequence(physicalDevice, device, compute_queue, 0));
    recorder->record<kp::OpSyncDevice>({fftBuffer})
        ->record<kp::OpSyncDevice>({FwBuffer})
        ->eval();
  }

  VkFFTConfiguration configuration = {};
  VkFFTApplication app = {};
  memset(&configuration, 0, sizeof(configuration));
  memset(&app, 0, sizeof(app));
  configuration.queue = &raw_compute_queue;
  configuration.fence = &fence;
  configuration.device = &raw_device;
  configuration.commandPool = &commandPool;
  configuration.physicalDevice = &raw_phydevice;
  configuration.isCompilerInitialized = false;
  configuration.FFTdim = 2;   // FFT dimension
  configuration.size[0] = mm; // FFT size X
  configuration.size[1] = nn; // FFT size Y
  configuration.numberBatches = 1;
  configuration.performR2C = 1;
  configuration.performDCT = 0;
  configuration.doublePrecision = 1;

  configuration.isInputFormatted = 1;
  configuration.inputBufferNum = 1;
  configuration.inputBufferSize = &inputBufferSize;
  configuration.inputBufferStride[0] = configuration.size[0];
  configuration.inputBufferStride[1] =
      configuration.size[0] * configuration.size[1];

  configuration.isOutputFormatted = 1;
  configuration.outputBufferNum = 1;
  configuration.outputBufferSize = &outputBufferSize;
  configuration.outputBufferStride[0] = configuration.size[0];
  configuration.outputBufferStride[1] =
      configuration.size[0] * configuration.size[1];

  configuration.bufferSize = &bufferSize;
  configuration.bufferNum = 1;

  VkFFTResult resFFT = initializeVkFFT(&app, configuration);
  VkFFTLaunchParams launchParams = {};
  launchParams.buffer = (VkBuffer *)fftBuffer->getPrimaryBuffer().get();
  launchParams.inputBuffer = (VkBuffer *)calReady->getPrimaryBuffer().get();
  launchParams.outputBuffer = (VkBuffer *)FwBuffer->getPrimaryBuffer().get();

  // NOTE: 执行VkFFT
  {
    VkResult res = VK_SUCCESS;
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = {};
    if (vkAllocateCommandBuffers(raw_device, &commandBufferAllocateInfo,
                                 &commandBuffer) != VK_SUCCESS)
      throw std::runtime_error(
          "onlyWFR::vkAllocateCommandBuffers call is failed");

    VkCommandBufferBeginInfo commandBufferBeginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) !=
        VK_SUCCESS)
      throw std::runtime_error("onlyWFR::vkBeginCommandBuffer call is failed");
    launchParams.commandBuffer = &commandBuffer;

    // NOTE: 这里可以添加多个FFT步骤
    if (VkFFTAppend(&app, -1, &launchParams) != VKFFT_SUCCESS)
      throw std::runtime_error("onlyWFR::VkFFTAppend is failed");
    res = vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    if (vkQueueSubmit(raw_compute_queue, 1, &submitInfo, fence) != VK_SUCCESS)
      throw std::runtime_error("onlyWFR::vkQueueSubmit call is failed");
    if (vkWaitForFences(raw_device, 1, &fence, VK_TRUE, 100000000000) !=
        VK_SUCCESS)
      throw std::runtime_error("onlyWFR::vkWaitForFences call is failed");
    if (vkResetFences(raw_device, 1, &fence) != VK_SUCCESS)
      throw std::runtime_error("onlyWFR::vkResetFences call is failed");
    vkFreeCommandBuffers(raw_device, commandPool, 1, &commandBuffer);
  }

  deleteVkFFT(&app);

  {
    const int HandleWidth = mm / 2;
    std::shared_ptr<kp::Algorithm> algo =
        mgr.algorithm<unsigned int, unsigned int>(
            {FwBuffer}, shader_step2, kp::Workgroup({1}),
            {HandleBlockSize, (unsigned)mm, (unsigned)nn,
             (unsigned)HandleWidth},
            {0});
    std::shared_ptr<kp::Sequence> recorder(
        new kp::Sequence(physicalDevice, device, compute_queue, 0));
    for (unsigned i = 0;
         i < (HandleWidth * nn + HandleBlockSize - 1) / HandleBlockSize; i++)
      recorder->record<kp::OpAlgoDispatch>(algo, std::vector<unsigned int>{i});
    // recorder->eval()->record<kp::OpSyncLocal>({FwBuffer})->eval();
    recorder->eval();
  }

  // DEBUG
  // {
  //   int width = 2 * sx + 1;
  //   int height = 2 * sy + 1;
  //   std::cout << "GaussianWindow Sum: "
  //             << GaussianWindow->data()[width * height] << std::endl;
  //   std::ofstream ofile("gpu_transform.txt", std::ios::trunc);
  //   for (int i = 0; i < height; i++) {
  //     for (int j = 0; j < width; j++) {
  //       ofile << GaussianWindow->data()[i * width + j] << ' ';
  //     }
  //     ofile << std::endl;
  //   }
  //   ofile.close();
  // }

  return {};
}

int main() {
  std::cout << "GPU Version - onlyWFR implementation" << std::endl;
  std::cout << "Please input any key to continue" << std::endl;
  std::cin.get();

  int img_width, img_height, img_comp;
  unsigned char *raw_image =
      stbi_load("/home/shenzhe/WorkSpace/vkWFR/matlab/smallpicture/000750.bmp",
                &img_width, &img_height, &img_comp, 1);

  std::cout << "Load Image Width: " << img_width << " Height: " << img_height
            << " Comp: " << img_comp << std::endl;
  assert(raw_image != nullptr);
  assert(img_comp == 3);

  // NOTE: 和Matlab比，由于超尾特性，这里的宽度要+1
  constexpr unsigned int roi_startX = 237, roi_startY = 293, roi_width = 101,
                         roi_height = 83;

  Eigen::Matrix<double, roi_height, roi_width> testImage;

  std::cout << "cropImage: Width: " << testImage.cols()
            << " Height: " << testImage.rows() << std::endl;

  // 调用onlyWFR函数
  auto start = std::chrono::high_resolution_clock::now();
  wfrResult result =
      onlyWFR({raw_image, (std::size_t)img_width * img_height}, img_width,
              img_height, {roi_startX, roi_startY, roi_width, roi_height},
              10,   // sigmax
              -0.5, // wxl
              0.1,  // wxi
              0.5,  // wxh
              10,   // sigmay
              -0.5, // wyl
              0.1,  // wyi
              0.5,  // wyh
              0.0); // thr (对于WFR不需要)
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

  stbi_image_free(raw_image);
  std::cout << "onlyWFR completed successfully!" << std::endl;
  std::cout << "Result dimensions: " << result.wx.rows() << "x"
            << result.wx.cols() << std::endl;
  std::cout << "Write result file" << std::endl;
  std::ofstream ofile("result.txt", std::ios::trunc);
  assert(ofile.is_open());
  ofile << result.r;
  ofile.close();
  return 0;
}
