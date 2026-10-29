#pragma once
// Little-endian binary (de)serialization helpers + CRC-32 (IEEE, zlib compatible).
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include "types.hpp"

namespace vf {

inline uint32_t crc32(const void* data, size_t n, uint32_t prev = 0) {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
      t[i] = c;
    }
    return t;
  }();
  uint32_t c = ~prev;
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
  return ~c;
}

inline bool host_is_little_endian() { const uint16_t x = 1; return *reinterpret_cast<const uint8_t*>(&x) == 1; }

class Writer {
 public:
  std::string buf;
  template <class T> void put(const T& v) {
    static_assert(std::is_trivially_copyable<T>::value, "POD only");
    buf.append(reinterpret_cast<const char*>(&v), sizeof(T));
  }
  void put_bytes(const void* p, size_t n) { buf.append(static_cast<const char*>(p), n); }
  void put_str(const std::string& s) { put<uint32_t>(uint32_t(s.size())); put_bytes(s.data(), s.size()); }
};

class Reader {
 public:
  Reader(const char* p, size_t n) : p_(p), end_(p + n) {}
  size_t remaining() const { return size_t(end_ - p_); }
  template <class T> T get() {
    static_assert(std::is_trivially_copyable<T>::value, "POD only");
    if (remaining() < sizeof(T)) throw CorruptionError("unexpected end of data");
    T v; std::memcpy(&v, p_, sizeof(T)); p_ += sizeof(T); return v;
  }
  void get_bytes(void* out, size_t n) {
    if (remaining() < n) throw CorruptionError("unexpected end of data");
    std::memcpy(out, p_, n); p_ += n;
  }
  std::string get_str() {
    uint32_t n = get<uint32_t>();
    if (remaining() < n) throw CorruptionError("string length exceeds buffer");
    std::string s(p_, n); p_ += n; return s;
  }
 private:
  const char* p_; const char* end_;
};

}  // namespace vf
