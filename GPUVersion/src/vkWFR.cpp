#include "FFT_R2C_2D.hpp"
#include "kompute/Tensor.hpp"
#include <shader/step1.hpp>
#include <shader/step2.hpp>
#include <shader/step3.hpp>
#include <shader/step4.hpp>
#include <shader/step5.hpp>
#include <shader/step6.hpp>
#include <shader/step7.hpp>
#include <stdexcept>
#include <vkWFR.hpp>

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
// WFR准备
const static std::vector<uint32_t> shader_step5(comp::STEP5_COMP_SPV.begin(),
                                                comp::STEP5_COMP_SPV.end());
// 复数矩阵元素乘法
const static std::vector<uint32_t> shader_step6(comp::STEP6_COMP_SPV.begin(),
                                                comp::STEP6_COMP_SPV.end());
// 脊线提取
const static std::vector<uint32_t> shader_step7(comp::STEP7_COMP_SPV.begin(),
                                                comp::STEP7_COMP_SPV.end());

// 工作组线程数
constexpr unsigned int HandleBlockSize = 32;

namespace kp {
class OpMemReset : public OpBase {
  std::vector<std::shared_ptr<TensorT<double>>> mTensors;

public:
  OpMemReset(const std::vector<std::shared_ptr<TensorT<double>>> buffObjs)
      : mTensors(buffObjs) {}

  ~OpMemReset() override {}

  void record(const vk::CommandBuffer &commandBuffer) override {
    for (auto &i : mTensors)
      commandBuffer.fillBuffer(*(i->getPrimaryBuffer()), 0, i->memorySize(), 0);
  }

  virtual void preEval(const vk::CommandBuffer &commandBuffer) override {}

  virtual void postEval(const vk::CommandBuffer &commandBuffer) override {}
};
} // namespace kp

vkWFR::vkWFR(int imgwidth_, int imgheight_, std::array<int, 4> ROI_, int sigmax,
             double wxl, double wxi, double wxh, int sigmay, double wyl,
             double wyi, double wyh, double thr)
    : imgWidth(imgwidth_), imgHeight(imgheight_), ROI(ROI_),
      sx(static_cast<int>(std::round(3 * sigmax))),
      sy(static_cast<int>(std::round(3 * sigmay))),
      mm(ROI[2] + 2 * static_cast<int>(std::round(3 * sigmax))),
      nn(ROI[3] + 2 * static_cast<int>(std::round(3 * sigmay))),
      cal_width(2 * static_cast<int>(std::round(3 * sigmax)) + 1),
      cal_height(2 * static_cast<int>(std::round(3 * sigmay)) + 1),
      sigmax(sigmax), sigmay(sigmay), wxl(wxl), wxi(wxi), wxh(wxh), wyl(wyl),
      wyi(wyi), wyh(wyh), thr(thr) {
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
    if (vkCreateInstance(&instanceCreateInfo, nullptr, &raw_instance) !=
        VK_SUCCESS)
      throw std::runtime_error(
          "vkWFR::constructor call vkCreateInstance failed");

    // 枚举物理设备
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(raw_instance, &deviceCount, nullptr);
    if (deviceCount == 0)
      throw std::runtime_error(
          "vkWFR::constructor No Vulkan physical devices found");

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(raw_instance, &deviceCount,
                               physicalDevices.data());

    // 选择第0个设备
    raw_phydevice = physicalDevices[0];

    // 准备设备扩展
    std::vector<const char *> deviceExtensions = {
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME};

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

    if (computeQueueFamilyIndex == UINT32_MAX)
      throw std::runtime_error(
          "vkWFR::constructor No compute queue family found");

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

    if (vkCreateDevice(raw_phydevice, &deviceCreateInfo, nullptr,
                       &raw_device) != VK_SUCCESS)
      throw std::runtime_error("Failed to create logical device");

    // 获取计算队列句柄
    vkGetDeviceQueue(raw_device, computeQueueFamilyIndex, 0,
                     &raw_compute_queue);
  }

  instance = std::make_shared<vk::Instance>(raw_instance);
  physicalDevice = std::make_shared<vk::PhysicalDevice>(raw_phydevice);
  device = std::make_shared<vk::Device>(raw_device);
  compute_queue = std::make_shared<vk::Queue>(raw_compute_queue);

  // 初始化kompute
  mgr = std::make_unique<kp::Manager>(instance, physicalDevice, device);

  // 初始化高斯窗
  {
    int workgroupNum =
        (cal_width * cal_height + HandleBlockSize - 1) / HandleBlockSize;
    GaussianWindow = mgr->tensorT<double>(
        std::vector<double>(cal_width * cal_height + workgroupNum));
    std::shared_ptr<kp::Algorithm> algo1 = mgr->algorithm<int, int>(
        {GaussianWindow}, shader_step3,
        kp::Workgroup({(unsigned int)workgroupNum, 1, 1}),
        {HandleBlockSize, cal_width, cal_height, sigmax, sigmay}, {});
    std::shared_ptr<kp::Sequence> recorder(
        new kp::Sequence(physicalDevice, device, compute_queue, 0));
    recorder->record<kp::OpAlgoDispatch>(algo1, std::vector<int>{});
    recorder->eval();
    // NOTE: 归一化
    std::shared_ptr<kp::Algorithm> algo2 = mgr->algorithm<int, int>(
        {GaussianWindow}, shader_step4,
        kp::Workgroup({(unsigned int)workgroupNum, 1, 1}),
        {HandleBlockSize, cal_width * cal_height, workgroupNum}, {});
    recorder->record<kp::OpAlgoDispatch>(algo2, std::vector<int>{});
    recorder->eval();
    ;
  }
  FfBuffer = mgr->tensorT<double>(std::vector<double>(mm * nn * 2));

  fft_r2c = std::make_unique<FFT_R2C_2D>(mm, nn, raw_instance, raw_phydevice,
                                         raw_device, computeQueueFamilyIndex);
  fft_c2c = std::make_unique<FFT_C2C_2D>(mm, nn, raw_instance, raw_phydevice,
                                         raw_device, computeQueueFamilyIndex);
}

std::vector<double> vkWFR::operator()(std::vector<unsigned char> image) {
  if (image.size() != imgWidth * imgHeight)
    throw std::runtime_error(
        "vkWFR::operator input image size not equal imgWidth*imgHeight");

  std::shared_ptr<kp::TensorT<double>> calReady =
      mgr->tensorT<double>(std::vector<double>(mm * nn, 0));
  {
    int workgroupNum =
        (imgWidth * imgHeight + HandleBlockSize - 1) / HandleBlockSize;
    // 这个用完后就没用了
    std::shared_ptr<kp::TensorT<unsigned char>> tensorIn =
        mgr->tensorT<unsigned char>(std::move(image));
    std::shared_ptr<kp::Algorithm> algo =
        mgr->algorithm<unsigned int, unsigned int>(
            {tensorIn, calReady}, shader_step1,
            kp::Workgroup({(unsigned int)workgroupNum, 1, 1}),
            {(unsigned int)imgWidth, (unsigned int)imgHeight, HandleBlockSize,
             (unsigned)ROI[0], (unsigned)ROI[1], (unsigned)ROI[2],
             (unsigned)ROI[3], (unsigned)mm, (unsigned)nn},
            {});
    std::shared_ptr<kp::Sequence> recorder(
        new kp::Sequence(physicalDevice, device, compute_queue, 0));
    recorder->record<kp::OpSyncDevice>({tensorIn, calReady});
    recorder->record<kp::OpAlgoDispatch>(algo, std::vector<unsigned int>{});
    recorder->eval();
  }

  // NOTE: 预计算输入频谱
  FFT_R2C_2D fft(mm, nn, raw_instance, raw_phydevice, raw_device,
                 computeQueueFamilyIndex);
  fft(calReady, FfBuffer);

  // NOTE: 完成Hermitian对称性
  {
    const int HandleWidth = mm / 2;
    const int workgroupNum =
        (HandleWidth * nn + HandleBlockSize - 1) / HandleBlockSize;
    std::shared_ptr<kp::Algorithm> algo =
        mgr->algorithm<unsigned int, unsigned int>(
            {FfBuffer}, shader_step2,
            kp::Workgroup({(unsigned int)workgroupNum, 1, 1}),
            {HandleBlockSize, (unsigned)mm, (unsigned)nn,
             (unsigned)HandleWidth},
            {});
    std::shared_ptr<kp::Sequence> recorder(
        new kp::Sequence(physicalDevice, device, compute_queue, 0));
    recorder->record<kp::OpAlgoDispatch>(algo, std::vector<unsigned int>{});
    recorder->eval();
  }

  std::shared_ptr<kp::TensorT<double>> result_ridge =
      mgr->tensorT<double>(std::vector<double>(ROI[2] * ROI[3], 0));
  std::shared_ptr<kp::TensorT<double>> w_expanded =
      mgr->tensorT<double>(std::vector<double>(mm * nn * 2, 0));
  std::shared_ptr<kp::Algorithm> algo_cal_w = mgr->algorithm<int, float>(
      {GaussianWindow, w_expanded}, shader_step5,
      kp::Workgroup(
          {(cal_width * cal_height + HandleBlockSize - 1) / HandleBlockSize, 1,
           1}),
      std::vector<int>{HandleBlockSize, cal_width, cal_height, (int)mm,
                       (int)nn},
      {0, 0});
  std::shared_ptr<kp::Algorithm> algo_cal_mulmatrix = mgr->algorithm<int, int>(
      {FfBuffer, w_expanded}, shader_step6,
      kp::Workgroup({(mm * nn + HandleBlockSize - 1) / HandleBlockSize, 1, 1}),
      std::vector<int>{HandleBlockSize, (int)mm * (int)nn}, {});
  std::shared_ptr<kp::Algorithm> algo_cal_ridge = mgr->algorithm<int, int>(
      {w_expanded, result_ridge}, shader_step7,
      kp::Workgroup(
          {(ROI[2] * ROI[3] + HandleBlockSize - 1) / HandleBlockSize, 1, 1}),
      std::vector<int>{HandleBlockSize, (int)mm, (int)nn, sx, sy, (int)ROI[2],
                       (int)ROI[3]},
      {});
  std::shared_ptr<kp::OpBase> memreset = std::make_shared<kp::OpMemReset>(
      std::vector<std::shared_ptr<kp::TensorT<double>>>{w_expanded});

  for (float wyt = wyl; wyt <= wyh + 1e-10; wyt += wyi) {
    for (float wxt = wxl; wxt <= wxh + 1e-10; wxt += wxi) {
      // NOTE: 计算基函数
      {
        std::shared_ptr<kp::Sequence> recorder(
            new kp::Sequence(physicalDevice, device, compute_queue, 0));
        recorder->record(memreset);
        recorder->record<kp::OpAlgoDispatch>(algo_cal_w,
                                             std::vector<float>{wxt, wyt});
        recorder->eval();
      }
      // NOTE: 计算频谱
      fft_c2c->forward(w_expanded);

      // NOTE: 矩阵元素乘法
      {
        std::shared_ptr<kp::Sequence> recorder(
            new kp::Sequence(physicalDevice, device, compute_queue, 0));
        recorder->record<kp::OpAlgoDispatch>(algo_cal_mulmatrix,
                                             std::vector<int>{});
        recorder->eval();
      }

      // NOTE: 转回时域
      fft_c2c->inverse(w_expanded);

      // NOTE: 提取脊频率
      {
        std::shared_ptr<kp::Sequence> recorder(
            new kp::Sequence(physicalDevice, device, compute_queue, 0));
        recorder->record<kp::OpAlgoDispatch>(algo_cal_ridge,
                                             std::vector<int>{});
        recorder->record<kp::OpSyncLocal>({result_ridge})->eval();
      }
    }
  }
  return result_ridge->vector();
}

vkWFR::~vkWFR() {}
