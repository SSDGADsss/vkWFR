#include "vkWFR.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

int main() {
  std::cout << "GPU Version - onlyWFR implementation" << std::endl;

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

  vkWFR wfrobj(img_width, img_height,
               {roi_startX, roi_startY, roi_width, roi_height},
               10,   // sigmax
               -0.5, // wxl
               0.1,  // wxi
               0.5,  // wxh
               10,   // sigmay
               -0.5, // wyl
               0.1,  // wyi
               0.5,  // wyh
               0.0); // thr (对于WFR不需要)

  std::vector<float> result;

  std::cout << "Please input any key to continue" << std::endl;
  std::cin.get();
  // 调用onlyWFR函数
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; i++)
    result = wfrobj(std::vector<unsigned char>(
        raw_image, raw_image + img_width * img_height));

  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

  stbi_image_free(raw_image);
  std::cout << "onlyWFR completed successfully!" << std::endl;
  std::cout << "Write result file" << std::endl;
  {
    std::ofstream ofile("gpu_transform.txt", std::ios::trunc);
    for (int i = 0; i < roi_height; i++) {
      for (int j = 0; j < roi_width; j++) {
        ofile << std::setw(14) << result[i * roi_width + j] << ' ';
      }
      ofile << std::endl;
    }
    ofile.close();
  }

  return 0;
}
