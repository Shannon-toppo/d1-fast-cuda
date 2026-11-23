// fp8 (e4m3) x fp8 GEMM for Ada (sm_89) with CUTLASS 2.x, as a plain DLL/shared library with a
// C ABI (loaded through ctypes): D (M, N) float32 = A (M, K) e4m3 x B (N, K) e4m3, transposed.
//
// - float32 accumulation with CUTLASS' staged accumulation (OpMultiplyAdd): the fp8 MMA's
//   partial sums are promoted to float32 regularly, as cuBLAS does.
// - The output is float32, unscaled, like `torch._scaled_mm(..., out_dtype=float32)` in
//   d1_fast_cuda.lm.linear: the per-row and per-channel scales stay outside, for torch.compile
//   to fuse into the next operation.
// - Built as C++17 without PyTorch headers: MSVC cannot compile CUTLASS' CuTe headers as C++20,
//   which PyTorch's headers require.
// - Several tile shapes, each data-parallel or Stream-K, by index (d1_fast_cuda/cutlass_fp8.py).

#include <cstddef>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#ifdef _WIN32
#define D1_EXPORT extern "C" __declspec(dllexport)
#else
#define D1_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementD = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

template <int TM, int TN, int TK, int WM, int WN, int WK, int Stages, bool StreamK>
struct Fp8Gemm {
  using Swizzle = typename std::conditional<StreamK, cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
                                            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>::type;
  using Gemm = cutlass::gemm::device::GemmUniversal<
      ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
      cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, WK>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombination<ElementD, 4, float, float>,
      Swizzle, Stages, 16, 16, cutlass::arch::OpMultiplyAdd>;

  static typename Gemm::Arguments args(int m, int n, int k, void const* a, void const* b, void* d, int sms) {
    typename Gemm::EpilogueOutputOp::Params epi(1.0f, 0.0f);
    cutlass::gemm::GemmCoord problem{m, n, k};
    if constexpr (StreamK) {
      return typename Gemm::Arguments(cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, epi, a, b, d, d,
                                      int64_t(m) * k, int64_t(n) * k, int64_t(m) * n, int64_t(m) * n,
                                      k, k, n, n, sms);
    } else {
      return typename Gemm::Arguments(cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, epi, a, b, d, d,
                                      int64_t(m) * k, int64_t(n) * k, int64_t(m) * n, int64_t(m) * n,
                                      k, k, n, n);
    }
  }

  static size_t workspace(int m, int n, int k, int sms) {
    return Gemm::get_workspace_size(args(m, n, k, nullptr, nullptr, nullptr, sms));
  }

  static int run(int m, int n, int k, void const* a, void const* b, void* d, void* ws, int sms, cudaStream_t s) {
    auto arguments = args(m, n, k, a, b, d, sms);
    Gemm gemm;
    cutlass::Status st = gemm.can_implement(arguments);
    if (st != cutlass::Status::kSuccess) return 1;
    st = gemm.initialize(arguments, ws, s);
    if (st != cutlass::Status::kSuccess) return 2;
    st = gemm.run(s);
    return st == cutlass::Status::kSuccess ? 0 : 3;
  }
};

struct Config {
  char const* name;
  size_t (*workspace)(int, int, int, int);
  int (*run)(int, int, int, void const*, void const*, void*, void*, int, cudaStream_t);
};

#define D1_CFG1(TM, TN, TK, WM, WN, WK, S, SK, NAME)                         \
  {NAME, &Fp8Gemm<TM, TN, TK, WM, WN, WK, S, SK>::workspace, &Fp8Gemm<TM, TN, TK, WM, WN, WK, S, SK>::run}
#define D1_CFG(TM, TN, TK, WM, WN, WK, S)                                                       \
  D1_CFG1(TM, TN, TK, WM, WN, WK, S, false, #TM "x" #TN "x" #TK "_" #WM "x" #WN "x" #WK "_s" #S), \
  D1_CFG1(TM, TN, TK, WM, WN, WK, S, true, #TM "x" #TN "x" #TK "_" #WM "x" #WN "x" #WK "_s" #S "_sk")

Config const kConfigs[] = {
    D1_CFG(16, 64, 128, 16, 32, 128, 5),
    D1_CFG(16, 128, 128, 16, 32, 128, 4),
    D1_CFG(32, 64, 128, 16, 32, 128, 5),
    D1_CFG(32, 128, 128, 32, 32, 128, 4),
    D1_CFG(64, 64, 128, 32, 32, 128, 4),
    D1_CFG(64, 64, 128, 32, 32, 128, 6),
    D1_CFG(64, 128, 128, 32, 64, 128, 3),
    D1_CFG(128, 64, 128, 64, 32, 128, 3),
    D1_CFG(128, 128, 64, 64, 64, 64, 4),
};
constexpr int kCount = int(sizeof(kConfigs) / sizeof(kConfigs[0]));

}  // namespace

D1_EXPORT int d1_fp8_count() { return kCount; }

D1_EXPORT char const* d1_fp8_name(int config) { return config >= 0 && config < kCount ? kConfigs[config].name : ""; }

D1_EXPORT size_t d1_fp8_workspace(int config, int m, int n, int k, int sms) {
  return kConfigs[config].workspace(m, n, k, sms);
}

// 0 on success; 1 cannot implement, 2 initialize failed, 3 launch failed, 4 bad config.
D1_EXPORT int d1_fp8_mm(int config, int m, int n, int k, void const* a, void const* b, void* d, void* workspace,
                        int sms, void* stream) {
  if (config < 0 || config >= kCount) return 4;
  return kConfigs[config].run(m, n, k, a, b, d, workspace, sms, static_cast<cudaStream_t>(stream));
}
