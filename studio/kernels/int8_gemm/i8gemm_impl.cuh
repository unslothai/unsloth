// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
//
// s8 GEMM with torchao's int8 dequant epilogue fused bit for bit: __fmul_rn / __fadd_rn (no FMA contraction),
// every PyTorch cast reproduced. A [M, K], W [N, K], out [M, N], all row-major.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

// clang-format off: the visitor headers need the GEMM headers first
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
// clang-format on

namespace ui8 {

using namespace cute;

// shared with the Python loader
enum Status : int {
  kOk = 0,
  kBadArgs = 1,
  kNoKernel = 2,
  kCannotImplement = 3,
  kInitFailed = 4,
  kLaunchFailed = 5,
};

enum Kind : int { kFp32 = 0, kBf16 = 1, kFp16 = 2 };

template <class T>
CUTLASS_DEVICE float round_to(float x);
template <>
CUTLASS_DEVICE float round_to<cutlass::bfloat16_t>(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}
template <>
CUTLASS_DEVICE float round_to<cutlass::half_t>(float x) {
  return __half2float(__float2half_rn(x));
}
template <>
CUTLASS_DEVICE float round_to<float>(float x) {
  return x;
}

template <class Out, class XsT, class WsT, bool Bias>
struct Dequant {
  // torch promotes int32 * bf16 to bf16; torchao widens an fp16 x_scale to fp32
  static constexpr bool kBf16Inter = std::is_same<XsT, cutlass::bfloat16_t>::value;
  static constexpr bool kRoundWs = std::is_same<WsT, Out>::value;

  CUTLASS_DEVICE static float scaled(float c, float xs, float ws) {
    float t = kBf16Inter ? round_to<Out>(__fmul_rn(round_to<Out>(c), xs)) : round_to<Out>(__fmul_rn(c, xs));
    float y = __fmul_rn(t, ws);
    return kRoundWs ? round_to<Out>(y) : y;
  }

  template <class T>
  struct Fn;

  template <int N>
  struct Fn<cutlass::Array<float, N>> {
    using Frag = cutlass::Array<float, N>;

    CUTLASS_DEVICE Frag operator()(Frag const& c, Frag const& xs, Frag const& ws) const {
      Frag y;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) {
        y[i] = round_to<Out>(scaled(c[i], xs[i], ws[i]));
      }
      return y;
    }

    CUTLASS_DEVICE Frag operator()(Frag const& c, Frag const& xs, Frag const& ws, Frag const& b) const {
      Frag y;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) {
        y[i] = round_to<Out>(__fadd_rn(scaled(c[i], xs[i], ws[i]), b[i]));
      }
      return y;
    }
  };
};

// diffusion_native_quant W8A8 Linear: fp32 epilogue, one cast, no bias
template <class Out>
struct NativeDequant {
  template <class T>
  struct Fn;

  template <int N>
  struct Fn<cutlass::Array<float, N>> {
    using Frag = cutlass::Array<float, N>;

    CUTLASS_DEVICE Frag operator()(Frag const& c, Frag const& xs, Frag const& ws) const {
      Frag y;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) {
        y[i] = round_to<Out>(__fmul_rn(__fmul_rn(c[i], xs[i]), ws[i]));
      }
      return y;
    }
  };
};

enum Epilogue : int { kTorchao = 0, kNative = 1 };

template <class Out, class XsT, class WsT, bool Bias, int Epi>
struct EpiFn;
template <class Out, class XsT, class WsT, bool Bias>
struct EpiFn<Out, XsT, WsT, Bias, kTorchao> {
  template <class T>
  using Fn = typename Dequant<Out, XsT, WsT, Bias>::template Fn<T>;
};
template <class Out, class XsT, class WsT, bool Bias>
struct EpiFn<Out, XsT, WsT, Bias, kNative> {
  static_assert(!Bias && std::is_same<XsT, float>::value && std::is_same<WsT, float>::value,
                "native epilogue: fp32 scales, no bias");
  template <class T>
  using Fn = typename NativeDequant<Out>::template Fn<T>;
};

template <int TBM_, int TBN_, int TBK_, int WM_, int WN_, int WK_, int Stages_, int Swizzle_ = 1>
struct Tile {
  static constexpr int TBM = TBM_, TBN = TBN_, TBK = TBK_;
  static constexpr int WM = WM_, WN = WN_, WK = WK_;
  static constexpr int Stages = Stages_;
  static constexpr int Swizzle = Swizzle_;
};

template <class ArchTag>
struct ArchTraits;
template <>
struct ArchTraits<cutlass::arch::Sm80> {
  using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
};
template <>
struct ArchTraits<cutlass::arch::Sm75> {
  using Inst = cutlass::gemm::GemmShape<8, 8, 16>;
};

template <class ArchTag, class T, class Out, class XsT, class WsT, bool Bias, int Epi = kTorchao>
struct FusedGemm {
  using ElementAcc = int32_t;
  using ElementCompute = float;
  static constexpr int kAlignAB = 16;
  static constexpr int kAlignC = 128 / cutlass::sizeof_bits<Out>::value;
  using TB = cutlass::gemm::GemmShape<T::TBM, T::TBN, T::TBK>;
  using Warp = cutlass::gemm::GemmShape<T::WM, T::WN, T::WK>;
  using Inst = typename ArchTraits<ArchTag>::Inst;
  static constexpr int kEVTStages = 1;

  using ThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, Out, kAlignC, kEVTStages>;
  using Acc = cutlass::epilogue::threadblock::VisitorAccFetch;
  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, XsT, Stride<_1, _0, int32_t>>;
  using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, WsT, Stride<_0, _1, int32_t>>;
  using BiasRow = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, Out, Stride<_0, _1, int32_t>>;
  using Compute = cutlass::epilogue::threadblock::VisitorCompute<EpiFn<Out, XsT, WsT, Bias, Epi>::template Fn, float,
                                                                 float, cutlass::FloatRoundStyle::round_to_nearest>;
  using Body = typename std::conditional<Bias, cutlass::epilogue::threadblock::Sm80EVT<Compute, Acc, XScale, WScale, BiasRow>,
                                         cutlass::epilogue::threadblock::Sm80EVT<Compute, Acc, XScale, WScale>>::type;
  using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
      ThreadMap, Out, cutlass::FloatRoundStyle::round_to_nearest, Stride<int64_t, _1, int64_t>>;
  using EVT = cutlass::epilogue::threadblock::Sm80EVT<Store, Body>;

  using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<T::Swizzle>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, kAlignAB, int8_t,
      cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, kAlignAB, Out, cutlass::layout::RowMajor,
      kAlignC, ElementAcc, ElementCompute, cutlass::arch::OpClassTensorOp, ArchTag, TB, Warp, Inst, EVT, Swizzle,
      T::Stages, cutlass::arch::OpMultiplyAddSaturate, kEVTStages>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static typename Body::Arguments body_args(const void* xs, const void* ws, const void* bias, int M, int N) {
    typename XScale::Arguments xa{static_cast<XsT const*>(xs), XsT(0), {_1{}, _0{}, M}};
    typename WScale::Arguments wa{static_cast<WsT const*>(ws), WsT(0), {_0{}, _1{}, N}};
    if constexpr (Bias) {
      typename BiasRow::Arguments ba{static_cast<Out const*>(bias), Out(0), {_0{}, _1{}, N}};
      return typename Body::Arguments{{}, xa, wa, ba, {}};
    } else {
      return typename Body::Arguments{{}, xa, wa, {}};
    }
  }

  static int run(const void* a, const void* w, const void* xs, const void* ws, const void* bias, void* out, int M,
                 int N, int K, int64_t lda, int64_t ldw, int64_t ldc, cudaStream_t stream) {
    typename EVT::Arguments cb{body_args(xs, ws, bias, M, N),
                               {static_cast<Out*>(out), {ldc, _1{}, int64_t(M) * ldc}}};
    cutlass::gemm::GemmCoord problem(M, N, K);
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, cb,
                                  const_cast<void*>(a), const_cast<void*>(w), nullptr, nullptr, int64_t(M) * lda,
                                  int64_t(N) * ldw, 0, 0, lda, ldw, 0, 0);
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return kCannotImplement;
    if (Gemm::get_workspace_size(args) != 0) return kCannotImplement;
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return kInitFailed;
    if (gemm.run(stream) != cutlass::Status::kSuccess) return kLaunchFailed;
    return cudaGetLastError() == cudaSuccess ? kOk : kLaunchFailed;
  }
};

using RunFn = int (*)(const void*, const void*, const void*, const void*, const void*, void*, int, int, int, int64_t,
                      int64_t, int64_t, cudaStream_t);

}  // namespace ui8
