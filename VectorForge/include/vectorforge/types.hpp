#pragma once
// Core vocabulary types for VectorForge.
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vf {

enum class Metric : uint8_t { L2 = 0, InnerProduct = 1, Cosine = 2 };

// Ordered key/value pairs (ordered => deterministic serialization).
using Metadata = std::vector<std::pair<std::string, std::string>>;

struct SearchResult {
  uint64_t id;
  float distance;  // L2: squared distance. Cosine: 1-cos. InnerProduct: -dot.
};

struct SearchStats {
  uint64_t dist_computations = 0;  // hardware-independent cost metric
  uint64_t strategy_used = 0;      // 0=exact 1=hnsw 2=hnsw-filtered 3=postfilter 4=bruteforce-filter
};

// Equality filter on a single metadata key. Empty key => no filter.
struct Filter {
  std::string key, value;
  bool active() const { return !key.empty(); }
};

struct Error : std::runtime_error { using std::runtime_error::runtime_error; };
struct InvalidArgument : Error { using Error::Error; };
struct CorruptionError : Error { using Error::Error; };
struct IoError : Error { using Error::Error; };

}  // namespace vf
