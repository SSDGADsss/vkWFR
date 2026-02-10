#ifndef STFT_H
#define STFT_H

#include <complex>
#include <fftw3.h>
#include <vector>

class STFT_1D {
  int L, R, N;
  double sigma;
  std::vector<double> window;
  double *in;
  fftw_complex *out;
  fftw_plan plan;

public:
  STFT_1D(int window_len, int hop, int fft_len, double sigma);
  ~STFT_1D();
  std::vector<std::vector<std::complex<double>>>
  compute(const std::vector<double> &signal);
};

#endif
