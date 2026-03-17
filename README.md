# vkWFR

基于Vulkan加速的加窗傅里叶脊算法

## 项目介绍

本项目在[WFR](https://ww2.mathworks.cn/matlabcentral/fileexchange/24852-windowed-fourier-transform-for-fringe-pattern-analysis)算法基础上，使用Vulkan API利用GPU进行加速，在`AMD Redeon 780M`核显上其性能相较于Matlab有着最高超过1956.85%的性能提升，并且具备和Matlab版本相近的精度。为解决高速高分辨条纹图的处理奠定了基础。

如下是高斯窗口大小为10x10的Benchmark，其中CPU版本和Matlab均运行在`AMD Ryzen 9700X`处理器上，Vulkan程序运行在`NVIDIA A10`显卡上

![Benchmark-10x10](https://github.com/SSDGADsss/vkWFR/blob/60a0dc2ec4c6af0533d90a9285fcca8fe0103b00/img/Gaussion%20Window%2010x10.png)

如下是高斯大小为10x10下，其相较于Matlab计算结果的误差分布图，其最大误差均在5e-4以下。

![ErrorDistribution](https://github.com/SSDGADsss/vkWFR/blob/90ceddb610a34dff0b7fb7ef4970f7e76d2a7066/img/Error%20Distribution.png)

## 编译

### 依赖环境

- 支持C++20的编译器
- [VkFFT](https://github.com/DTolm/VkFFT)
- [Kompute](https://github.com/KomputeProject/kompute/)
- [HDF5](https://github.com/HDFGroup/hdf5)
- [STB](https://github.com/nothings/stb)

### 构建

```
git clone https://github.com/SSDGADsss/vkWFR.git
cd vkWFR && mkdir build && cd build
cmake .. -GCMAKE_BUILD_TYPE=Release
make -j
```

## 使用

### CMake Subproject

```cmake
add_subdirectory(GPUVersion)

add_executable("Your Project")

target_link_libraries("Your Project" PUBLIC VkFFT)
```

### Run Benchmark

在构建目录下有`GPUBenchmark`和`CPUBenchmark`两个可执行文件，运行即可获得性能数据
