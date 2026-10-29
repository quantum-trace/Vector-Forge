#pragma once
// Distance kernels. Written with independent accumulators so that GCC/Clang
// auto-vectorize them without -ffast-math (results are deterministic per build).
#include <cmath>
#include <cstddef>
#include "types.hpp"

namespace vf {

inline float dot(const float* a, const float* b, size_t d) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  size_t i = 0;
  for (; i + 8 <= d; i += 8) {
    s0 += a[i] * b[i];         s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4]; s5 += a[i + 5] * b[i + 5];
    s6 += a[i + 6] * b[i + 6]; s7 += a[i + 7] * b[i + 7];
  }
  float s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
  for (; i < d; ++i) s += a[i] * b[i];
  return s;
}

inline float l2_sq(const float* a, const float* b, size_t d) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  size_t i = 0;
  for (; i + 8 <= d; i += 8) {
    float t0 = a[i] - b[i], t1 = a[i + 1] - b[i + 1], t2 = a[i + 2] - b[i + 2], t3 = a[i + 3] - b[i + 3];
    float t4 = a[i + 4] - b[i + 4], t5 = a[i + 5] - b[i + 5], t6 = a[i + 6] - b[i + 6], t7 = a[i + 7] - b[i + 7];
    s0 += t0 * t0; s1 += t1 * t1; s2 += t2 * t2; s3 += t3 * t3;
    s4 += t4 * t4; s5 += t5 * t5; s6 += t6 * t6; s7 += t7 * t7;
  }
  float s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
  for (; i < d; ++i) { float t = a[i] - b[i]; s += t * t; }
  return s;
}

inline float norm(const float* a, size_t d) { return std::sqrt(dot(a, a, d)); }

// Cosine distance assumes both inputs are unit-normalized (the Collection
// normalizes on ingest and on query). Reference version for tests handles raw input.
inline float cosine_unit(const float* a, const float* b, size_t d) { return 1.0f - dot(a, b, d); }
inline float neg_dot(const float* a, const float* b, size_t d) { return -dot(a, b, d); }

inline float cosine_reference(const float* a, const float* b, size_t d) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < d; ++i) { ab += double(a[i]) * b[i]; aa += double(a[i]) * a[i]; bb += double(b[i]) * b[i]; }
  if (aa == 0 || bb == 0) return 1.0f;  // convention: undefined similarity => maximal distance 1
  return float(1.0 - ab / std::sqrt(aa * bb));
}

using DistFn = float (*)(const float*, const float*, size_t);
inline DistFn dist_fn(Metric m) {
  switch (m) {
    case Metric::L2: return l2_sq;
    case Metric::InnerProduct: return neg_dot;
    case Metric::Cosine: return cosine_unit;
  }
  return l2_sq;
}

}  // namespace vf
