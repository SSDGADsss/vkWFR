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

  fftwf_complex *pre_handle_in = nullptr, *pre_handle_out = nullptr;

  fftwf_plan pre_plan;

  std::vector<float> calReady;       // mm*nn*2
  std::vector<float> GaussianWindow; // cal_width*cal_height
  std::vector<float> result_ridge;

  std::vector<std::array<float, 2>> calFreqList;

  struct Parallel_Freq {
    fftwf_complex *freq_handle_in;
    fftwf_complex *freq_handle_out;
    fftwf_complex *ifreq_handle_in;
    fftwf_complex *ifreq_handle_out;
    fftwf_plan freq_plan;
    fftwf_plan ifreq_plan;
  };

  std::vector<Parallel_Freq> ParallelPool;

public:
  cpuWFR(int imgwidth, int imgheight, std::array<int, 4> ROI, int sigmax,
         float wxl, float wxi, float wxh, int sigmay, float wyl, float wyi,
         float wyh);
  ~cpuWFR();
  std::vector<float> operator()(std::vector<unsigned char> image);
};

#endif
