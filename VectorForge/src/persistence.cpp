#include "vectorforge/persistence.hpp"
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include "vectorforge/io.hpp"

namespace vf {

static const char kWalMagic[8] = {'V', 'F', 'W', 'A', 'L', '0', '0', '1'};

static void write_all(int fd, const char* p, size_t n) {
  while (n) {
    ssize_t w = ::write(fd, p, n);
    if (w < 0) { if (errno == EINTR) continue; throw IoError(std::string("write failed: ") + std::strerror(errno)); }
    p += w; n -= size_t(w);
  }
}

bool file_exists(const std::string& path) { struct stat st; return ::stat(path.c_str(), &st) == 0; }

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw IoError("cannot open " + path);
  std::ostringstream ss; ss << f.rdbuf();
  return ss.str();
}

void fsync_dir(const std::string& dir) {
  int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
  if (fd < 0) throw IoError("cannot open dir " + dir);
  int rc = ::fsync(fd);
  ::close(fd);
  if (rc != 0) throw IoError("fsync(dir) failed");
}

void atomic_write_file(const std::string& dir, const std::string& name, const std::string& data) {
  const std::string tmp = dir + "/" + name + ".tmp", dst = dir + "/" + name;
  int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) throw IoError("cannot create " + tmp);
  try {
    write_all(fd, data.data(), data.size());
    if (::fsync(fd) != 0) throw IoError("fsync failed");
  } catch (...) { ::close(fd); ::unlink(tmp.c_str()); throw; }
  ::close(fd);
  if (::rename(tmp.c_str(), dst.c_str()) != 0) throw IoError("rename failed");
  fsync_dir(dir);
}

void Wal::swap_fd(int fd) {
  std::unique_lock<std::mutex> l(sync_mu_);
  sync_cv_.wait(l, [&] { return !syncing_; });   // never close a descriptor a leader is fdatasync-ing
  if (fd_ >= 0) ::close(fd_);
  fd_ = fd;
  // A new log starts only after a snapshot covering every record appended so far was published
  // (checkpoint) or at open/recovery, so everything appended up to now is durable.
  durable_lsn_ = appended_lsn_;
  failed_ = false;
  l.unlock();
  sync_cv_.notify_all();
}

void Wal::create(const std::string& path, uint64_t base_lsn) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
  if (fd < 0) throw IoError("cannot create " + path);
  Writer w; w.put_bytes(kWalMagic, 8); w.put<uint64_t>(base_lsn);
  try {
    write_all(fd, w.buf.data(), w.buf.size());
    if (::fsync(fd) != 0) throw IoError("fsync failed");
  } catch (...) { ::close(fd); throw; }
  bytes_ = w.buf.size(); unsynced_ = 0; appended_lsn_ = base_lsn;
  swap_fd(fd);
}

void Wal::open_append(const std::string& path, uint64_t valid_bytes) {
  if (::truncate(path.c_str(), off_t(valid_bytes)) != 0) throw IoError("truncate failed");
  int fd = ::open(path.c_str(), O_WRONLY | O_APPEND);
  if (fd < 0) throw IoError("cannot open " + path);
  if (::fsync(fd) != 0) { ::close(fd); throw IoError("fsync failed"); }
  bytes_ = valid_bytes; unsynced_ = 0;
  swap_fd(fd);
}

void Wal::append(uint64_t lsn, uint8_t op, const std::string& body, SyncMode mode, size_t every_n) {
  if (fd_ < 0) throw IoError("wal not open");
  Writer pl; pl.put<uint64_t>(lsn); pl.put<uint8_t>(op); pl.put_bytes(body.data(), body.size());
  Writer rec;
  rec.put<uint32_t>(uint32_t(pl.buf.size()));
  rec.put<uint32_t>(crc32(pl.buf.data(), pl.buf.size()));
  rec.put_bytes(pl.buf.data(), pl.buf.size());
  write_all(fd_, rec.buf.data(), rec.buf.size());  // single write(2): one record, one syscall
  bytes_ += rec.buf.size();
  { std::lock_guard<std::mutex> g(sync_mu_); appended_lsn_ = lsn; }
  if (mode == SyncMode::Always) wait_durable(lsn);
  else if (mode == SyncMode::EveryN && ++unsynced_ >= every_n) { unsynced_ = 0; wait_durable(lsn); }
  // SyncMode::Group: the caller invokes wait_durable(lsn) after leaving its critical section.
}

// Leader/follower group commit. At most one thread runs fdatasync at a time; it makes durable every
// record appended before it started (target), and wakes all waiters whose LSN is now covered.
void Wal::wait_durable(uint64_t lsn) {
  std::unique_lock<std::mutex> l(sync_mu_);
  while (durable_lsn_ < lsn) {
    if (failed_) throw IoError("wal poisoned by an earlier fdatasync failure; reopen the collection");
    if (!syncing_) {
      syncing_ = true;
      const uint64_t target = appended_lsn_;
      const int fd = fd_;
      l.unlock();
      const int rc = ::fdatasync(fd);
      l.lock();
      syncing_ = false;
      if (rc != 0) { failed_ = true; sync_cv_.notify_all(); throw IoError("fdatasync failed"); }
      if (target > durable_lsn_) durable_lsn_ = target;
      sync_cv_.notify_all();
    } else {
      sync_cv_.wait(l);
    }
  }
}

void Wal::sync() {
  uint64_t t;
  { std::lock_guard<std::mutex> g(sync_mu_); t = appended_lsn_; }
  wait_durable(t);
}

void Wal::close() {
  std::unique_lock<std::mutex> l(sync_mu_);
  sync_cv_.wait(l, [&] { return !syncing_; });
  if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

Wal::ReadResult Wal::read_all(const std::string& path) {
  const std::string data = read_file(path);
  ReadResult rr;
  rr.file_bytes = data.size();
  if (data.size() < 16 || std::memcmp(data.data(), kWalMagic, 8) != 0) throw CorruptionError("wal: bad header");
  std::memcpy(&rr.base_lsn, data.data() + 8, 8);
  size_t pos = 16;
  rr.valid_bytes = pos;
  uint64_t prev_lsn = rr.base_lsn;
  while (pos < data.size()) {
    if (data.size() - pos < 8) { rr.torn_tail = true; break; }
    uint32_t len, crc;
    std::memcpy(&len, data.data() + pos, 4);
    std::memcpy(&crc, data.data() + pos + 4, 4);
    if (len < 9 || len > (1u << 30) || data.size() - pos - 8 < len) {
      // length runs past EOF => torn append. (An absurd length mid-file is corruption.)
      if (data.size() - pos - 8 < len && len <= (1u << 30)) { rr.torn_tail = true; break; }
      throw CorruptionError("wal: invalid record length at offset " + std::to_string(pos));
    }
    const char* pl = data.data() + pos + 8;
    const bool at_eof = (pos + 8 + len == data.size());
    if (crc32(pl, len) != crc) {
      if (at_eof) { rr.torn_tail = true; break; }
      throw CorruptionError("wal: checksum mismatch at offset " + std::to_string(pos));
    }
    Record r;
    std::memcpy(&r.lsn, pl, 8);
    r.op = uint8_t(pl[8]);
    r.body.assign(pl + 9, len - 9);
    if (r.lsn <= prev_lsn) throw CorruptionError("wal: non-monotonic LSN");
    prev_lsn = r.lsn;
    rr.records.push_back(std::move(r));
    pos += 8 + len;
    rr.valid_bytes = pos;
  }
  return rr;
}

DirLock::DirLock(const std::string& dir) {
  const std::string p = dir + "/LOCK";
  fd_ = ::open(p.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd_ < 0) throw IoError("cannot open lock file");
  if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) { ::close(fd_); fd_ = -1; throw IoError("collection is locked by another process/instance"); }
}
DirLock::~DirLock() { if (fd_ >= 0) { ::flock(fd_, LOCK_UN); ::close(fd_); } }

}  // namespace vf
