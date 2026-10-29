#pragma once
// Contiguous vector storage. Internal ids are dense uint32 indices; external ids
// (uint64 labels) are mapped by the Collection. Storage is row-major for cache locality.
#include <cstdint>
#include <vector>
#include "types.hpp"

namespace vf {

class VectorStore {
 public:
  explicit VectorStore(size_t dim) : dim_(dim) {}
  size_t dim() const { return dim_; }
  size_t size() const { return labels_.size(); }
  size_t live() const { return live_; }
  const float* vec(uint32_t i) const { return data_.data() + size_t(i) * dim_; }
  uint64_t label(uint32_t i) const { return labels_[i]; }
  bool deleted(uint32_t i) const { return deleted_[i] != 0; }
  const Metadata& meta(uint32_t i) const { return meta_[i]; }

  uint32_t add(uint64_t label, const float* v, Metadata md) {
    if (labels_.size() >= 0xFFFFFFFEu) throw Error("store: capacity (2^32-2 vectors) exceeded");
    data_.insert(data_.end(), v, v + dim_);
    labels_.push_back(label);
    deleted_.push_back(0);
    meta_.push_back(std::move(md));
    ++live_;
    return uint32_t(labels_.size() - 1);
  }
  void mark_deleted(uint32_t i) { if (!deleted_[i]) { deleted_[i] = 1; --live_; } }
  void reserve(size_t n) { data_.reserve(n * dim_); labels_.reserve(n); deleted_.reserve(n); meta_.reserve(n); }
  size_t memory_bytes() const {
    size_t m = data_.size() * sizeof(float) + labels_.size() * 8 + deleted_.size() + meta_.size() * sizeof(Metadata);
    for (auto& md : meta_) for (auto& kv : md) m += kv.first.size() + kv.second.size() + 2 * sizeof(std::string);
    return m;
  }
 private:
  size_t dim_;
  size_t live_ = 0;
  std::vector<float> data_;
  std::vector<uint64_t> labels_;
  std::vector<uint8_t> deleted_;
  std::vector<Metadata> meta_;
};

}  // namespace vf
