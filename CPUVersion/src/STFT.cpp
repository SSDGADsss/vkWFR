#include <STFT.hpp>

STFT_1D::STFT_1D(int window_len, int hop, int fft_len, double sigma)
    : L(window_len), R(hop), N(fft_len), sigma(sigma) {
  // 参数检查
  if (N < L)
    N = L;
  // 分配内存
  window.resize(L);
  in = (double *)fftw_malloc(sizeof(double) * N);
  out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
  plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_MEASURE);
  // 计算高斯窗
  double mean = (L - 1) / 2.0;
  for (int i = 0; i < L; ++i) {
    window[i] = std::exp(-0.5 * std::pow((i - mean) / sigma, 2));
  }
  // 可选：归一化窗，使能量为1
  // double sum = 0; for (auto w: window) sum += w*w; sum = sqrt(sum); for
  // (auto& w: window) w /= sum;
}

STFT_1D::~STFT_1D() {
  fftw_destroy_plan(plan);
  fftw_free(in);
  fftw_free(out);
}

// 处理整个信号，返回频谱矩阵：每帧有 N/2+1 个复数
std::vector<std::vector<std::complex<double>>>
STFT_1D::compute(const std::vector<double> &signal) {
  int total_samples = signal.size();
  int num_frames = (total_samples - L) / R + 1;
  std::vector<std::vector<std::complex<double>>> spectrogram(num_frames);
  // 对于每一帧
  for (int frame = 0; frame < num_frames; ++frame) {
    int start = frame * R;
    // 复制加窗数据到 in，其余补零
    for (int i = 0; i < L; ++i) {
      in[i] = signal[start + i] * window[i];
    }
    for (int i = L; i < N; ++i) {
      in[i] = 0.0;
    }
    // 执行 FFT
    fftw_execute(plan);
    // 转换到 std::complex
    int nbins = N / 2 + 1;
    std::vector<std::complex<double>> frame_spectrum(nbins);
    for (int k = 0; k < nbins; ++k) {
      frame_spectrum[k] = std::complex<double>(out[k][0], out[k][1]);
    }
    spectrogram[frame] = std::move(frame_spectrum);
  }
  return spectrogram;
}
