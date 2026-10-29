#pragma once
// Durability primitives: write-ahead log, atomic file publication, directory lock.
//
// WAL file layout
//   header : "VFWAL001" (8) | u64 base_lsn      -- base_lsn = LSN of the snapshot it extends
//   record : u32 payload_len | u32 crc32(payload) | payload
//   payload: u64 lsn | u8 op | body
// Recovery policy
//   * A record that is cut short by EOF, or whose CRC fails *and which ends at EOF*,
//     is a torn tail from a crash mid-append: it is discarded (and the file truncated).
//   * A CRC failure followed by more data is mid-log corruption: recovery refuses
//     (CorruptionError) instead of silently dropping acknowledged writes.
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include "types.hpp"

namespace vf {

// Always : append + fdatasync inside the writer's critical section (simplest, serialises fsyncs).
// Group  : append inside the critical section, fdatasync *outside* it with group commit: concurrent
//          writers share one fdatasync (a "leader" syncs up to the newest appended LSN). A write is
//          acknowledged only after it is durable, exactly as with Always, but throughput scales with
//          the number of concurrent writers instead of being capped at 1/fsync-latency.
// EveryN : fdatasync every N records (acknowledged writes may be lost on power failure).
// None   : never sync explicitly (OS decides).
enum class SyncMode { Always, EveryN, None, Group };

enum WalOp : uint8_t { OP_UPSERT = 1, OP_DELETE = 2 };

class Wal {
 public:
  struct Record { uint64_t lsn; uint8_t op; std::string body; };
  struct ReadResult {
    uint64_t base_lsn = 0;
    std::vector<Record> records;
    uint64_t valid_bytes = 0, file_bytes = 0;
    bool torn_tail = false;
  };

  Wal() = default;
  Wal(const Wal&) = delete;
  Wal& operator=(const Wal&) = delete;
  ~Wal() { close(); }

  void create(const std::string& path, uint64_t base_lsn);  // fresh log (caller publishes atomically)
  void open_append(const std::string& path, uint64_t valid_bytes);  // truncates to valid_bytes first
  void append(uint64_t lsn, uint8_t op, const std::string& body, SyncMode mode, size_t every_n);
  void sync();                        // make everything appended so far durable
  void wait_durable(uint64_t lsn);    // group commit: block until `lsn` is durable (thread-safe)
  void close();
  bool is_open() const { return fd_ >= 0; }
  uint64_t bytes() const { return bytes_; }

  static ReadResult read_all(const std::string& path);

 private:
  void swap_fd(int fd);  // replace the descriptor once no sync is in flight
  int fd_ = -1;
  size_t unsynced_ = 0;
  uint64_t bytes_ = 0;
  std::mutex sync_mu_;
  std::condition_variable sync_cv_;
  bool syncing_ = false, failed_ = false;
  uint64_t appended_lsn_ = 0, durable_lsn_ = 0;
};

void fsync_dir(const std::string& dir);
// Write data to dir/name atomically: tmp file + fsync + rename + fsync(dir).
void atomic_write_file(const std::string& dir, const std::string& name, const std::string& data);
std::string read_file(const std::string& path);
bool file_exists(const std::string& path);

// Exclusive advisory lock on dir/LOCK, released on destruction or process death.
class DirLock {
 public:
  explicit DirLock(const std::string& dir);
  ~DirLock();
  DirLock(const DirLock&) = delete;
  DirLock& operator=(const DirLock&) = delete;
 private:
  int fd_ = -1;
};

}  // namespace vf
