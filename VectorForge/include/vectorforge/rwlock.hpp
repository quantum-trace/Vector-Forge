#pragma once
// Reader/writer lock with a selectable fairness policy.
//  * writer_preference=true : once a writer is waiting, new readers queue behind it
//    (bounded write latency; readers may briefly stall behind a write).
//  * writer_preference=false: readers are admitted while no writer is *active* (the glibc
//    pthread_rwlock default). Maximizes read throughput but can starve writers indefinitely.
// Satisfies SharedMutex requirements used by std::shared_lock / std::unique_lock.
#include <condition_variable>
#include <mutex>

namespace vf {

class RWLock {
 public:
  void set_writer_preference(bool v) { writer_pref_ = v; }
  void lock_shared() {
    std::unique_lock<std::mutex> l(m_);
    cv_.wait(l, [&] { return !writer_active_ && (!writer_pref_ || writers_waiting_ == 0); });
    ++readers_;
  }
  void unlock_shared() {
    std::unique_lock<std::mutex> l(m_);
    if (--readers_ == 0) { l.unlock(); cv_.notify_all(); }
  }
  void lock() {
    std::unique_lock<std::mutex> l(m_);
    ++writers_waiting_;
    cv_.wait(l, [&] { return !writer_active_ && readers_ == 0; });
    --writers_waiting_;
    writer_active_ = true;
  }
  void unlock() {
    { std::lock_guard<std::mutex> l(m_); writer_active_ = false; }
    cv_.notify_all();
  }
 private:
  std::mutex m_;
  std::condition_variable cv_;
  int readers_ = 0, writers_waiting_ = 0;
  bool writer_active_ = false, writer_pref_ = true;
};

}  // namespace vf
