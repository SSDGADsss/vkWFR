#ifndef CPU_WFR_H
#define CPU_WFR_H

#include <array>
#include <fftw3.h>
#include <vector>

class cpuWFR {
  const unsigned int mm, nn;
  const int sx, sy;
  const int cal_width, cal_height;
  const int sigmax, sigmay;
  const float wxl, wxi, wxh, wyl, wyi, wyh;

  const int imgWidth, imgHeight;
  const std::array<int, 4> ROI;

  fftw_complex *pre_handle_in = nullptr, *pre_handle_out = nullptr;
  fftw_complex *freq_handle_in = nullptr, *freq_handle_out = nullptr;
  fftw_complex *ifreq_handle_in = nullptr, *ifreq_handle_out = nullptr;

  fftw_plan pre_plan, freq_plan, ifreq_plan;

  std::vector<double> calReady;       // mm*nn*2
  std::vector<double> GaussianWindow; // cal_width*cal_height
  std::vector<double> result_ridge;

public:
  cpuWFR(int imgwidth, int imgheight, std::array<int, 4> ROI, int sigmax,
         float wxl, float wxi, float wxh, int sigmay, float wyl, float wyi,
         float wyh);
  ~cpuWFR();
  std::vector<double> operator()(std::vector<unsigned char> image);
};

#endif
