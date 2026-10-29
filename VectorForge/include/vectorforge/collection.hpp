#pragma once
// Collection: the public engine API. Combines VectorStore + HNSW + metadata index,
// provides thread-safe access (reader/writer lock) and optional crash-safe persistence.
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include "hnsw.hpp"
#include "persistence.hpp"
#include "rwlock.hpp"
#include "store.hpp"
#include "types.hpp"

namespace vf {

struct CollectionConfig {
  size_t dim = 0;
  Metric metric = Metric::L2;
  HnswParams hnsw;
  bool writer_preference = true;  // lock fairness policy (runtime-only; not persisted)
  // Automatic compaction (runtime-only). When > 0, the index is rebuilt without tombstones as soon as
  // tombstones >= auto_compact_min and tombstones/(live+tombstones) >= auto_compact_ratio. The rebuild runs
  // inside the writer that triggers it (an exclusive-lock latency spike proportional to the live set).
  // After a compaction the internal graph differs from what WAL replay would rebuild: answers stay
  // logically correct but ANN results may differ marginally after a restart.
  double auto_compact_ratio = 0.0;
  size_t auto_compact_min = 1000;
};

struct PersistOptions {
  SyncMode sync = SyncMode::Always;  // acknowledge a write only after it is durable
  size_t sync_every = 1000;          // used by SyncMode::EveryN
};

enum class FilterStrategy { Auto, Traversal, PostFilter, BruteForce };

struct SearchParams {
  size_t k = 10;
  size_t ef = 64;                       // efSearch
  Filter filter;
  FilterStrategy strategy = FilterStrategy::Auto;
  bool exact = false;                   // force exhaustive scan (ground truth / small data)
  size_t brute_force_threshold = 4096;  // Auto: scan posting list if it has <= this many live matches
  double post_filter_fraction = 0.25;   // Auto: post-filter when matches >= this fraction of live (conservative; see paper 5.6)
};

struct CollectionStats {
  size_t dim = 0, live = 0, tombstones = 0;
  size_t vector_bytes = 0, graph_bytes = 0, index_bytes = 0;
  uint64_t last_lsn = 0, wal_bytes = 0;
  int max_level = -1;
};

struct RecoveryInfo {
  bool recovered = false;
  uint64_t snapshot_lsn = 0, wal_records_replayed = 0, wal_records_skipped = 0, wal_bytes_truncated = 0;
  double snapshot_load_ms = 0, wal_replay_ms = 0;
};

class Collection {
 public:
  explicit Collection(CollectionConfig cfg);  // in-memory only
  // Opens (or creates) a persistent collection in `dir`. For an existing collection the
  // dim/metric in `cfg` must match the stored ones; HNSW parameters come from the snapshot.
  static std::unique_ptr<Collection> open(const std::string& dir, const CollectionConfig& cfg,
                                          PersistOptions opts = {}, RecoveryInfo* info = nullptr);
  ~Collection();
  Collection(const Collection&) = delete;
  Collection& operator=(const Collection&) = delete;

  void upsert(uint64_t id, const float* v, size_t dim, const Metadata& md = {});
  bool erase(uint64_t id);
  std::vector<SearchResult> search(const float* q, size_t dim, const SearchParams& sp, SearchStats* st = nullptr) const;
  bool get(uint64_t id, std::vector<float>* out, Metadata* md = nullptr) const;
  bool contains(uint64_t id) const;
  size_t size() const;
  CollectionStats stats() const;
  const CollectionConfig& config() const { return cfg_; }

  void checkpoint();  // persistent only: atomic snapshot + WAL rotation
  void sync();        // persistent only: force WAL to disk
  void compact();     // rebuild graph without tombstones (logical state unchanged)
  size_t compactions() const;  // number of compactions performed (manual + automatic)
  // Test hook: validate internal invariants (graph + id map + posting lists).
  bool check_invariants(std::string* why = nullptr) const;

 private:
  struct Posting { std::vector<uint32_t> ids; size_t live = 0; };
  static std::string pkey(const std::string& k, const std::string& v) { return k + '\x1f' + v; }

  void validate(const float* v, size_t dim, bool for_query) const;
  void prepare(const float* v, std::vector<float>& out) const;  // copy + normalize for cosine
  void apply_upsert(uint64_t id, const float* raw, const Metadata& md);
  bool apply_erase(uint64_t id);
  void rebuild_aux();  // id map + postings from store
  void tombstone(uint32_t iid);
  void compact_locked();
  void maybe_auto_compact();
  std::string serialize_snapshot(uint64_t lsn) const;
  void load_snapshot(const std::string& data);
  bool matches(uint32_t iid, const Filter& f) const;

  CollectionConfig cfg_;
  std::unique_ptr<VectorStore> store_;
  std::unique_ptr<Hnsw> hnsw_;
  std::unordered_map<uint64_t, uint32_t> id_map_;
  std::unordered_map<std::string, Posting> postings_;
  mutable RWLock mu_;

  // persistence
  bool persistent_ = false;
  std::string dir_;
  PersistOptions popts_;
  std::unique_ptr<DirLock> lock_;
  Wal wal_;
  uint64_t last_lsn_ = 0;
  std::mutex ckpt_mu_;
  size_t compactions_ = 0;
};

}  // namespace vf
