#include "FFT_R2C_2D.hpp"
#include "kompute/Core.hpp"
#include "kompute/Sequence.hpp"
#include "kompute/Tensor.hpp"
#include "kompute/operations/OpAlgoDispatch.hpp"
#include "kompute/operations/OpSyncLocal.hpp"
#include <memory>
#include <shader/step1.hpp>
#include <shader/step2.hpp>
#include <shader/step3.hpp>
#include <shader/step4.hpp>
#include <shader/step5.hpp>
#include <shader/step6.hpp>
#include <shader/step7.hpp>
#include <shader/step8.hpp>
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
// 归并结果
const static std::vector<uint32_t> shader_step8(comp::STEP8_COMP_SPV.begin(),
                                                comp::STEP8_COMP_SPV.end());

// 工作组线程数
constexpr unsigned int HandleBlockSize = 32;

// 频率计算并行数量
constexpr unsigned int FreqCalParallelNumber = 10;

namespace kp {
class OpMemReset : public OpBase {
  std::vector<std::shared_ptr<TensorT<float>>> mTensors;

public:
  OpMemReset(const std::vector<std::shared_ptr<TensorT<float>>> buffObjs)
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
             float wxl, float wxi, float wxh, int sigmay, float wyl, float wyi,
             float wyh, float thr)
    : imgWidth(imgwidth_), imgHeight(imgheight_), ROI(ROI_),
      sx(static_cast<int>(std::round(3 * sigmax))),
      sy(static_cast<int>(std::round(3 * sigmay))),
      mm(ROI[2] + 2 * static_cast<int>(std::round(3 * sigmax))),
      nn(ROI[3] + 2 * static_cast<int>(std::round(3 * sigmay))),
      cal_width(2 * static_cast<int>(std::round(3 * sigmax)) + 1),
      cal_height(2 * static_cast<int>(std::round(3 * sigmay)) + 1),
      sigmax(sigmax), sigmay(sigmay), wxl(wxl), wxi(wxi), wxh(wxh), wyl(wyl),
      wyi(wyi), wyh(wyh), thr(thr), FreqCalPool(FreqCalParallelNumber) {
  {
    // 初始化频率计算表
    calFreqList.reserve(((wyh + 1e-10 - wyl) / wyi + 1) *
                        ((wyh + 1e-10 - wyl) / wyi + 1));
    for (float wyt = wyl; wyt <= wyh + 1e-10; wyt += wyi)
      for (float wxt = wxl; wxt <= wxh + 1e-10; wxt += wxi)
        calFreqList.push_back({wxt, wyt, 1});
    while (calFreqList.size() % FreqCalParallelNumber != 0) {
      calFreqList.push_back({0, 0, 0});
    }

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

  {
    // NOTE: 初始化高斯窗
    int workgroupNum =
        (cal_width * cal_height + HandleBlockSize - 1) / HandleBlockSize;
    GaussianWindow = mgr->tensorT<float>(
        std::vector<float>(cal_width * cal_height + workgroupNum));
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
  FfBuffer = mgr->tensorT<float>(std::vector<float>(mm * nn * 2));
  result_ridge = mgr->tensorT<float>(std::vector<float>(ROI[2] * ROI[3]));
  calReady = mgr->tensorT<float>(std::vector<float>(mm * nn));
  tensorIn = mgr->tensorT<unsigned char>(
      std::vector<unsigned char>(imgWidth * imgHeight));

  algo_cal_init = mgr->algorithm<unsigned int, unsigned int>(
      {tensorIn, calReady}, shader_step1,
      kp::Workgroup(
          {(unsigned int)((imgWidth * imgHeight + HandleBlockSize - 1) /
                          HandleBlockSize),
           1, 1}),
      {(unsigned int)imgWidth, (unsigned int)imgHeight, HandleBlockSize,
       (unsigned)ROI[0], (unsigned)ROI[1], (unsigned)ROI[2], (unsigned)ROI[3],
       (unsigned)mm, (unsigned)nn},
      {});
  algo_cal_hermitian = mgr->algorithm<unsigned int, unsigned int>(
      {FfBuffer}, shader_step2,
      kp::Workgroup({(unsigned int)((mm / 2 * nn + HandleBlockSize - 1) /
                                    HandleBlockSize),
                     1, 1}),
      {HandleBlockSize, (unsigned)mm, (unsigned)nn, (unsigned)mm / 2}, {});
  initMem = std::make_shared<kp::OpMemReset>(
      std::vector<std::shared_ptr<kp::TensorT<float>>>{result_ridge, calReady});

  {
    recorder_init = std::shared_ptr<kp::Sequence>(
        new kp::Sequence(physicalDevice, device, compute_queue, 0));
    recorder_init->record(initMem);
    recorder_init->record<kp::OpSyncDevice>(
        {algo_cal_init->getMemObjects()[0]});
    recorder_init->record<kp::OpAlgoDispatch>(algo_cal_init,
                                              std::vector<unsigned int>{});
  }
  {
    recorder_hermitian = std::shared_ptr<kp::Sequence>(
        new kp::Sequence(physicalDevice, device, compute_queue, 0));
    recorder_hermitian->record<kp::OpAlgoDispatch>(algo_cal_hermitian,
                                                   std::vector<unsigned int>{});
  }

  fft_r2c = std::make_unique<FFT_R2C_2D>(mm, nn, calReady, FfBuffer,
                                         raw_instance, raw_phydevice,
                                         raw_device, computeQueueFamilyIndex);
  for (int i = 0; i < FreqCalParallelNumber; i++) {
    auto &freqcal = FreqCalPool[i];
    freqcal.w_expanded = mgr->tensorT<float>(std::vector<float>(mm * nn * 2));
    freqcal.result = mgr->tensorT<float>(std::vector<float>(mm * nn));
    freqcal.memreset = std::make_shared<kp::OpMemReset>(
        std::vector<decltype(freqcal.w_expanded)>{freqcal.w_expanded,
                                                  freqcal.result});
    freqcal.algo_cal_w = mgr->algorithm<int, float>(
        {GaussianWindow, freqcal.w_expanded}, shader_step5,
        kp::Workgroup(
            {(cal_width * cal_height + HandleBlockSize - 1) / HandleBlockSize,
             1, 1}),
        std::vector<int>{HandleBlockSize, cal_width, cal_height, (int)mm,
                         (int)nn},
        {0, 0, 1});
    freqcal.algo_cal_mulmatrix = mgr->algorithm<int, int>(
        {FfBuffer, freqcal.w_expanded}, shader_step6,
        kp::Workgroup(
            {(mm * nn + HandleBlockSize - 1) / HandleBlockSize, 1, 1}),
        std::vector<int>{HandleBlockSize, (int)mm * (int)nn}, {});
    freqcal.algo_cal_ridge = mgr->algorithm<int, int>(
        {freqcal.w_expanded, freqcal.result}, shader_step7,
        kp::Workgroup(
            {(ROI[2] * ROI[3] + HandleBlockSize - 1) / HandleBlockSize, 1, 1}),
        std::vector<int>{HandleBlockSize, (int)mm, (int)nn, sx, sy, (int)ROI[2],
                         (int)ROI[3]},
        {});
    {
      freqcal.recorder_mulmatrix = std::shared_ptr<kp::Sequence>(
          new kp::Sequence(physicalDevice, device, compute_queue, 0));
      freqcal.recorder_mulmatrix->record<kp::OpAlgoDispatch>(
          freqcal.algo_cal_mulmatrix, std::vector<int>{});
    }
    {
      freqcal.recorder_cal_ridge = std::shared_ptr<kp::Sequence>(
          new kp::Sequence(physicalDevice, device, compute_queue, 0));
      freqcal.recorder_cal_ridge->record<kp::OpAlgoDispatch>(
          freqcal.algo_cal_ridge, std::vector<int>{});
      // freqcal.recorder_cal_ridge->record<kp::OpSyncLocal>({result_ridge});
    }
    {
      freqcal.recorder_base_func = std::shared_ptr<kp::Sequence>(
          new kp::Sequence(physicalDevice, device, compute_queue, 0));
      freqcal.recorder_base_func->record(freqcal.memreset);
      freqcal.recorder_base_func->record<kp::OpAlgoDispatch>(
          freqcal.algo_cal_w);
    }
    freqcal.fft_c2c = std::make_unique<FFT_C2C_2D>(
        mm, nn, freqcal.w_expanded, raw_instance, raw_phydevice, raw_device,
        computeQueueFamilyIndex);
  }
  std::vector<std::shared_ptr<kp::Memory>> mergeMemory(FreqCalParallelNumber +
                                                       1);
  for (int i = 0; i < FreqCalParallelNumber; i++)
    mergeMemory[i] = FreqCalPool[i].result;
  mergeMemory[FreqCalParallelNumber] = result_ridge;

  algo_cal_merge = mgr->algorithm<int, int>(
      mergeMemory, shader_step8,
      kp::Workgroup{(mm * nn + HandleBlockSize - 1) / HandleBlockSize, 1, 1},
      std::vector<int>{HandleBlockSize, (int)(mm * nn)}, std::vector<int>{});
  recorder_mergeResult = std::shared_ptr<kp::Sequence>(
      new kp::Sequence(physicalDevice, device, compute_queue, 0));
  recorder_mergeResult->record<kp::OpAlgoDispatch>(algo_cal_merge);

  recorder_allfinish = std::shared_ptr<kp::Sequence>(
      new kp::Sequence(physicalDevice, device, compute_queue, 0));
  recorder_allfinish->record<kp::OpSyncLocal>({result_ridge});
}

std::vector<float> vkWFR::operator()(std::vector<unsigned char> image) {
  if (image.size() != imgWidth * imgHeight)
    throw std::runtime_error(
        "vkWFR::operator input image size not equal imgWidth*imgHeight");

  // NOTE: 准备数据
  algo_cal_init->getMemObjects()[0]->setData(image.data(), image.size());
  recorder_init->eval();

  // NOTE: 预计算输入频谱
  (*fft_r2c)();

  // NOTE: 完成Hermitian对称性
  recorder_hermitian->eval();

  for (int i = 0; i < calFreqList.size() / FreqCalParallelNumber; i++) {
    for (int j = 0; j < FreqCalParallelNumber; j++) {
      // NOTE: 计算基函数
      FreqCalPool[j].algo_cal_w->setPushConstants(
          calFreqList[i * FreqCalParallelNumber + j].data(), 3, sizeof(float));
      FreqCalPool[j].recorder_base_func->evalAsync();
    }

    for (int j = 0; j < FreqCalParallelNumber; j++)
      FreqCalPool[j].recorder_base_func->evalAwait();

    // NOTE: 计算频谱
    for (int j = 0; j < FreqCalParallelNumber; j++)
      FreqCalPool[j].fft_c2c->asyncForward();

    for (int j = 0; j < FreqCalParallelNumber; j++)
      FreqCalPool[j].fft_c2c->waitAsync();

    // NOTE: 矩阵元素乘法
    for (int j = 0; j < FreqCalParallelNumber; j++)
      FreqCalPool[j].recorder_mulmatrix->evalAsync();

    for (int j = 0; j < FreqCalParallelNumber; j++)
      FreqCalPool[j].recorder_mulmatrix->evalAwait();

    // NOTE: 转回时域
    for (int j = 0; j < FreqCalParallelNumber; j++)
      FreqCalPool[j].fft_c2c->asyncInverse();

    for (int j = 0; j < FreqCalParallelNumber; j++)
      FreqCalPool[j].fft_c2c->waitAsync();

    // NOTE: 提取脊频率
    for (int j = 0; j < FreqCalParallelNumber; j++)
      FreqCalPool[j].recorder_cal_ridge->evalAsync();

    for (int j = 0; j < FreqCalParallelNumber; j++)
      FreqCalPool[j].recorder_cal_ridge->evalAwait();

    // NOTE: 归并
    recorder_mergeResult->eval();
  }
  recorder_allfinish->eval();
  std::vector<float> result(ROI[2] * ROI[3]);
  memcpy(result.data(), result_ridge->data(), ROI[2] * ROI[3] * sizeof(float));
  return result;
}

vkWFR::~vkWFR() {}
