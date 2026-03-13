#include <cmath>
#include <cpuWFR.hpp>
#include <cstring>
#include <fftw3.h>

cpuWFR::cpuWFR(int imgwidth, int imgheight, std::array<int, 4> ROI, int sigmax,
               float wxl, float wxi, float wxh, int sigmay, float wyl,
               float wyi, float wyh)
    : imgWidth(imgwidth), imgHeight(imgheight), ROI(ROI), sigmax(sigmax),
      sigmay(sigmay), wxl(wxl), wxi(wxi), wxh(wxh), wyl(wyl), wyi(wyi),
      wyh(wyh), cal_width(2 * static_cast<int>(std::round(3 * sigmax)) + 1),
      cal_height(2 * static_cast<int>(std::round(3 * sigmay)) + 1),
      sx(static_cast<int>(std::round(3 * sigmax))),
      sy(static_cast<int>(std::round(3 * sigmay))),
      mm(ROI[2] + 2 * static_cast<int>(std::round(3 * sigmax))),
      nn(ROI[3] + 2 * static_cast<int>(std::round(3 * sigmay))) {

  pre_handle_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * mm * nn);
  pre_handle_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * mm * nn);
  freq_handle_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * mm * nn);
  freq_handle_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * mm * nn);
  ifreq_handle_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * mm * nn);
  ifreq_handle_out =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * mm * nn);

  pre_plan = fftw_plan_dft_2d(mm, nn, pre_handle_in, pre_handle_out,
                              FFTW_FORWARD, FFTW_ESTIMATE);
  freq_plan = fftw_plan_dft_2d(mm, nn, freq_handle_in, freq_handle_out,
                               FFTW_FORWARD, FFTW_ESTIMATE);
  ifreq_plan = fftw_plan_dft_2d(mm, nn, ifreq_handle_in, ifreq_handle_out,
                                FFTW_BACKWARD, FFTW_ESTIMATE);

  calReady.resize(mm * nn * 2);
  GaussianWindow.resize(cal_height * cal_width);
  result_ridge.resize(ROI[2] * ROI[3]);

  // 计算窗口中心坐标（从0开始索引）
  double center_x = (cal_width - 1) / 2.0;
  double center_y = (cal_height - 1) / 2.0;

  // 预计算常数
  const double sigma_x_sq = static_cast<double>(sigmax) * sigmax;
  const double sigma_y_sq = static_cast<double>(sigmay) * sigmay;
  const double two_sigma_x_sq = 2.0 * sigma_x_sq;
  const double two_sigma_y_sq = 2.0 * sigma_y_sq;

  // 生成高斯窗并计算平方和
  // WARN: 这里是行优先，只是具备对称性而已
  double sum_of_squares = 0.0;

  for (int y = 0; y < cal_height; ++y) {
    const double dy = static_cast<double>(y) - center_y;
    const double dy_sq = dy * dy;
    const double y_term = dy_sq / two_sigma_y_sq;

    for (int x = 0; x < cal_width; ++x) {
      const double dx = static_cast<double>(x) - center_x;
      const double dx_sq = dx * dx;
      const double x_term = dx_sq / two_sigma_x_sq;

      // 计算高斯值：exp(-(dx²/(2σx²) + dy²/(2σy²)))
      const double exponent = -(x_term + y_term);
      const double gaussian_value = std::exp(exponent);

      // 存储到GaussianWindow中
      GaussianWindow[y * cal_width + x] = gaussian_value;

      // 累加平方和用于归一化
      sum_of_squares += gaussian_value * gaussian_value;
    }
  }

  // 归一化：使得窗口的L2范数为1（即平方和为1）
  double norm_factor = std::sqrt(sum_of_squares);

  // 应用归一化
  for (int i = 0; i < cal_height * cal_width; ++i)
    GaussianWindow[i] /= norm_factor;
}

cpuWFR::~cpuWFR() {
  fftw_destroy_plan(pre_plan);
  fftw_destroy_plan(freq_plan);
  fftw_destroy_plan(ifreq_plan);

  fftw_free(pre_handle_in);
  fftw_free(freq_handle_in);
  fftw_free(ifreq_handle_in);
  fftw_free(pre_handle_out);
  fftw_free(freq_handle_out);
  fftw_free(ifreq_handle_out);
}

std::vector<double> cpuWFR::operator()(std::vector<unsigned char> image) {
  memset(pre_handle_in, 0, sizeof(fftw_complex) * mm * nn);

  for (int y = 0; y < ROI[3]; y++)
    for (int x = 0; x < ROI[2]; x++)
      pre_handle_in[x * nn + y][0] =
          image[(y + ROI[1]) * imgWidth + ROI[0] + x];

  // 此时，列优先
  fftw_execute(pre_plan);

  memset(result_ridge.data(), 0, sizeof(double) * ROI[2] * ROI[3]);
  for (double wyt = wyl; wyt <= wyh + 1e-10; wyt += wyi) {
    for (double wxt = wxl; wxt <= wxh + 1e-10; wxt += wxi) {
      memset(freq_handle_in, 0, sizeof(double) * mm * nn * 2);
      for (int i = 0; i < cal_width; ++i) {
        for (int j = 0; j < cal_height; ++j) {
          const double impl = wxt * (i - (cal_width - 1) / 2) +
                              wyt * (j - (cal_height - 1) / 2);
          freq_handle_in[i * nn + j][0] =
              GaussianWindow[i * cal_height + j] * cos(impl);
          freq_handle_in[i * nn + j][1] =
              GaussianWindow[i * cal_height + j] * sin(impl);
        }
      }

      fftw_execute(freq_plan);

      for (int i = 0; i < mm * nn; i++) {
        ifreq_handle_in[i][0] = (pre_handle_out[i][0] * freq_handle_out[i][0] -
                                 pre_handle_out[i][1] * freq_handle_out[i][1]);
        ifreq_handle_in[i][1] = (pre_handle_out[i][0] * freq_handle_out[i][1] +
                                 pre_handle_out[i][1] * freq_handle_out[i][0]);
      }

      fftw_execute(ifreq_plan);
      for (int i = 0; i < mm * nn; i++) {
        ifreq_handle_out[i][0] /= mm * nn;
        ifreq_handle_out[i][1] /= mm * nn;
      }

      for (int i = 0; i < ROI[2]; i++)
        for (int j = 0; j < ROI[3]; j++)
          result_ridge[i * ROI[3] + j] = std::max(
              std::sqrt(ifreq_handle_out[(sx + i) * nn + (sy + j)][0] *
                            ifreq_handle_out[(sx + i) * nn + (sy + j)][0] +
                        ifreq_handle_out[(sx + i) * nn + (sy + j)][1] *
                            ifreq_handle_out[(sx + i) * nn + (sy + j)][1]),
              result_ridge[i * ROI[3] + j]);
    }
  }

  std::vector<double> result(result_ridge.size());
  // 这里转回行优先
  for (int i = 0; i < ROI[2]; i++)
    for (int j = 0; j < ROI[3]; j++)
      result[j * ROI[2] + i] = result_ridge[i * ROI[3] + j];

  return result;
}
