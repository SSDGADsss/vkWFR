#include "vkWFR.hpp"
#include <H5Cpp.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

int main() {
  std::cout << "GPU Version - onlyWFR implementation" << std::endl;

  int img_width, img_height, img_comp;
  // unsigned char *raw_image =
  //     stbi_load("/home/shenzhe/WorkSpace/vkWFR/matlab/bigPicture/000001.bmp",
  //               &img_width, &img_height, &img_comp, 1);
  unsigned char *raw_image =
      stbi_load("/home/shenzhe/WorkSpace/vkWFR/matlab/Phase/128x128_1.bmp",
                &img_width, &img_height, &img_comp, 1);

  std::cout << "Load Image Width: " << img_width << " Height: " << img_height
            << " Comp: " << img_comp << std::endl;
  assert(raw_image != nullptr);
  assert(img_comp == 3);

  // NOTE: 和Matlab比，由于超尾特性，这里的宽度要+1
  // constexpr unsigned int roi_startX = 123, roi_startY = 198, roi_width = 142,
  //                        roi_height = 166;

  vkWFR wfrobj(img_width, img_height,
               // {roi_startX, roi_startY, roi_width, roi_height},
               {0, 0, img_width, img_height},
               20,  // sigmax
               -1,  // wxl
               0.1, // wxi
               1,   // wxh
               20,  // sigmay
               -1,  // wyl
               0.1, // wyi
               1    // wyh
  );

  std::vector<float> result;

  std::cout << "Please input any key to continue" << std::endl;
  std::cin.get();
  // 调用onlyWFR函数
  auto start = std::chrono::steady_clock::now();
  result = wfrobj(std::vector<unsigned char>(
      raw_image, raw_image + img_width * img_height));

  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

  stbi_image_free(raw_image);
  std::cout << "onlyWFR completed successfully!" << std::endl;
  std::cout << "Write result file" << std::endl;

  try {
    H5::H5File file("phase-2.h5", H5F_ACC_TRUNC);

    // 创建数据空间：2D数组，尺寸为[img_height][img_width]
    hsize_t dims[2] = {static_cast<hsize_t>(img_height),
                       static_cast<hsize_t>(img_width)};
    H5::DataSpace dataspace(2, dims);

    // 创建浮点数据类型
    H5::FloatType datatype(H5::PredType::NATIVE_FLOAT);
    datatype.setOrder(H5T_ORDER_LE);

    // 创建数据集
    H5::DataSet dataset = file.createDataSet("result", datatype, dataspace);

    // 写入数据（result已经是行优先顺序）
    dataset.write(result.data(), H5::PredType::NATIVE_FLOAT);

    // 关闭所有资源（RAII会自动处理，但显式关闭更安全）
    dataset.close();
    dataspace.close();
    file.close();

    std::cout << "HDF5 file 'gpu_transform.h5' written successfully."
              << std::endl;
  } catch (H5::Exception &e) {
    std::cerr << "HDF5 error: " << e.getDetailMsg() << std::endl;
    return -1;
  } catch (std::exception &e) {
    std::cerr << "Standard error: " << e.what() << std::endl;
    return -1;
  }
  // {
  //   std::ofstream ofile("gpu_transform.txt", std::ios::trunc);
  //   for (int i = 0; i < img_height; i++) {
  //     for (int j = 0; j < img_width; j++) {
  //       ofile << std::setw(14) << result[i * img_width + j] << ' ';
  //     }
  //     ofile << std::endl;
  //   }
  //   ofile.close();
  // }

  return 0;
}
