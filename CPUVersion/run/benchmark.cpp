#include "cpuWFR.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

void RunBenchmark(std::string filepath, bool smallPara) {
  int img_width, img_height, img_comp;
  unsigned char *raw_image =
      stbi_load(filepath.c_str(), &img_width, &img_height, &img_comp, 1);
  if (raw_image == nullptr)
    throw std::runtime_error("Open image failed");

  std::unique_ptr<cpuWFR> algo;
  if (smallPara)
    algo = std::make_unique<cpuWFR>(
        img_width, img_height, std::array<int, 4>{0, 0, img_width, img_height},
        10,   // sigmax
        -0.5, // wxl
        0.1,  // wxi
        0.5,  // wxh
        10,   // sigmay
        -0.5, // wyl
        0.1,  // wyi
        0.5   // wyh
    );
  else
    algo = std::make_unique<cpuWFR>(
        img_width, img_height, std::array<int, 4>{0, 0, img_width, img_height},
        20,  // sigmax
        -1,  // wxl
        0.1, // wxi
        1,   // wxh
        20,  // sigmay
        -1,  // wyl
        0.1, // wyi
        1    // wyh
    );

  std::cout << "Test Start: " << filepath << " with "
            << (smallPara ? "small" : "big") << " param" << std::endl;
  std::vector<double> result;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; i++)
    result = (*algo)(std::vector<unsigned char>(
        raw_image, raw_image + img_width * img_height));

  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << ">>> Spend Time Pre Image: " << duration.count() / 100 << " ms"
            << std::endl;
  stbi_image_free(raw_image);
}

int main() {
  std::cout << "vkWFR Benchmark - CPU" << std::endl;
  RunBenchmark("Phase/128x128_1.bmp", true);
  RunBenchmark("Phase/128x128_1.bmp", false);
  RunBenchmark("Phase/256x256_1.bmp", true);
  RunBenchmark("Phase/256x256_1.bmp", false);
  RunBenchmark("Phase/512x512_1.bmp", true);
  RunBenchmark("Phase/512x512_1.bmp", false);
  RunBenchmark("Phase/1024x1024_1.bmp", true);
  RunBenchmark("Phase/1024x1024_1.bmp", false);
  RunBenchmark("Phase/2048x2048_1.bmp", true);
  RunBenchmark("Phase/2048x2048_1.bmp", false);
  RunBenchmark("Phase/4096x4096_1.bmp", true);
  RunBenchmark("Phase/4096x4096_1.bmp", false);
  return 0;
}
