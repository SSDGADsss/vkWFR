#include <Eigen/Eigen>
#include <Eigen/src/Core/util/Constants.h>
#include <chrono>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <fstream>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

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
wfrResult onlyWFR(Eigen::MatrixXd f, double sigmax, double wxl, double wxi,
                  double wxh, double sigmay, double wyl, double wyi, double wyh,
                  double thr) {
  // 注释：onlyWFR函数 - 窗口傅里叶脊提取算法
  // 该函数用于从单个条纹图案中提取相位信息

  wfrResult result;

  // 步骤1：计算窗口半尺寸（类似MATLAB中的round(3*sigmax)）
  int sx = static_cast<int>(std::round(3 * sigmax));
  int sy = static_cast<int>(std::round(3 * sigmay));

  // 步骤2：获取输入图像尺寸
  int m = f.rows();
  int n = f.cols();

  // 步骤3：计算扩展尺寸（用于卷积）
  int mm = m + 2 * sx;
  int nn = n + 2 * sy;

  // 步骤4：将输入转换为复数矩阵
  MatrixXcd f_complex = f.cast<Complex>();

  // 步骤5：扩展输入图像到尺寸[mm nn]
  MatrixXcd f_expanded = fexpand(f_complex, mm, nn);

  // 步骤6：预计算输入图像的频谱
  MatrixXcd Ff = fft2(f_expanded);

  // 步骤7：创建网格坐标（用于生成窗口）
  Eigen::MatrixXd x = Eigen::MatrixXd::Zero(2 * sx + 1, 2 * sy + 1);
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(2 * sx + 1, 2 * sy + 1);

  for (int i = -sx; i <= sx; ++i) {
    for (int j = -sy; j <= sy; ++j) {
      x(i + sx, j + sy) = i;
      y(i + sx, j + sy) = j;
    }
  }

  // 步骤8：生成高斯窗口w0
  Eigen::MatrixXd w0 = Eigen::MatrixXd::Zero(2 * sx + 1, 2 * sy + 1);
  double sigmax2 = 2 * sigmax * sigmax;
  double sigmay2 = 2 * sigmay * sigmay;

  for (int i = 0; i < 2 * sx + 1; ++i) {
    for (int j = 0; j < 2 * sy + 1; ++j) {
      double xi = x(i, j);
      double yi = y(i, j);
      w0(i, j) = std::exp(-(xi * xi) / sigmax2 - (yi * yi) / sigmay2);
    }
  }

  // 步骤9：对窗口进行L2归一化
  double w0_norm = std::sqrt(w0.array().square().sum());
  w0 = w0 / w0_norm;

  // 步骤10：初始化结果矩阵
  result.wx = Eigen::MatrixXd::Zero(m, n);
  result.wy = Eigen::MatrixXd::Zero(m, n);
  result.phase = Eigen::MatrixXd::Zero(m, n);
  result.r = Eigen::MatrixXd::Zero(m, n);

  // 步骤11：遍历频率范围
  for (double wyt = wyl; wyt <= wyh + 1e-10; wyt += wyi) {
    for (double wxt = wxl; wxt <= wxh + 1e-10; wxt += wxi) {
      // 步骤12：创建WFT基函数 w = w0 * exp(j*wxt*x + j*wyt*y)
      MatrixXcd w = MatrixXcd::Zero(2 * sx + 1, 2 * sy + 1);

      for (int i = 0; i < 2 * sx + 1; ++i) {
        for (int j = 0; j < 2 * sy + 1; ++j) {
          Complex exponent(0.0, wxt * x(i, j) + wyt * y(i, j));
          w(i, j) = w0(i, j) * std::exp(exponent);
        }
      }

      // 步骤13：扩展窗口到尺寸[mm nn]
      MatrixXcd w_expanded = fexpand(w, mm, nn);

      // 步骤14：计算窗口的频谱
      MatrixXcd Fw = fft2(w_expanded);

      // 步骤15：实现WFT：conv2(f,w) = ifft2(Ff * Fw)
      MatrixXcd sf_freq = Ff.array() * Fw.array();
      MatrixXcd sf = ifft2(sf_freq);

      // 步骤16：裁剪到原始尺寸
      MatrixXcd sf_cropped = sf.block(sx, sy, m, n);

      // 步骤17：计算幅度并确定需要更新的位置
      Eigen::MatrixXd sf_abs = Eigen::MatrixXd::Zero(m, n);
      for (int i = 0; i < m * n; ++i) {
        sf_abs(i) = std::abs(sf_cropped(i));
      }

      // 步骤18：创建更新掩码（t = (abs(sf) > r)）
      Eigen::MatrixXd t = Eigen::MatrixXd::Zero(m, n);
      for (int i = 0; i < m * n; ++i) {
        t(i) = (sf_abs(i) > result.r(i)) ? 1.0 : 0.0;
      }

      // 步骤19：更新ridge值
      result.r =
          result.r.array() * (1.0 - t.array()) + sf_abs.array() * t.array();

      // 步骤20：更新wx频率
      result.wx = result.wx.array() * (1.0 - t.array()) + wxt * t.array();

      // 步骤21：更新wy频率
      result.wy = result.wy.array() * (1.0 - t.array()) + wyt * t.array();

      // 步骤22：更新相位
      Eigen::MatrixXd sf_phase = Eigen::MatrixXd::Zero(m, n);
      for (int i = 0; i < m * n; ++i)
        sf_phase(i) = std::arg(sf_cropped(i));
      result.phase = result.phase.array() * (1.0 - t.array()) +
                     sf_phase.array() * t.array();
    }
  }
  return result;
}

int main() {
  std::cout << "CPU Version - onlyWFR implementation" << std::endl;
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

  constexpr int roi_startX = 237, roi_startY = 293, roi_width = 101,
                roi_height = 82;

  Eigen::Matrix<double, roi_height, roi_width> testImage;

  std::cout << "cropImage: Width: " << testImage.cols()
            << " Height: " << testImage.rows() << std::endl;

  for (int y = 0; y < roi_height; y++) {
    const unsigned char *ptrx =
        raw_image + (y + roi_startY) * img_width + roi_startX;
    for (int x = 0; x < roi_width; x++, ptrx++) {
      testImage(y, x) = *ptrx;
    }
  }

  stbi_image_free(raw_image);

  // 调用onlyWFR函数
  auto start = std::chrono::high_resolution_clock::now();
  wfrResult result = onlyWFR(testImage,
                             10.0, // sigmax
                             -0.5, // wxl
                             0.1,  // wxi
                             0.5,  // wxh
                             10.0, // sigmay
                             -0.5, // wyl
                             0.1,  // wyi
                             0.5,  // wyh
                             0.0); // thr (对于WFR不需要)
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

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
