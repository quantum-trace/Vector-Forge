#include "vectorforge/collection.hpp"
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <queue>

namespace vf {

namespace {
const char kSnapMagic[8] = {'V', 'F', 'S', 'N', 'A', 'P', '0', '1'};
constexpr uint32_t kSnapVersion = 1;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
}  // namespace

Collection::Collection(CollectionConfig cfg) : cfg_(cfg) {
  if (!host_is_little_endian()) throw Error("VectorForge requires a little-endian host");
  if (cfg.dim == 0 || cfg.dim > 65536) throw InvalidArgument("dim must be in [1,65536]");
  store_ = std::make_unique<VectorStore>(cfg.dim);
  hnsw_ = std::make_unique<Hnsw>(store_.get(), cfg.metric, cfg.hnsw);
  mu_.set_writer_preference(cfg.writer_preference);
}

Collection::~Collection() {
  if (persistent_) { try { wal_.sync(); } catch (...) {} }
}

// ---------- validation & helpers ----------

void Collection::validate(const float* v, size_t dim, bool for_query) const {
  if (v == nullptr) throw InvalidArgument("null vector");
  if (dim != cfg_.dim) throw InvalidArgument("dimension mismatch: expected " + std::to_string(cfg_.dim) + ", got " + std::to_string(dim));
  double ss = 0;
  for (size_t i = 0; i < dim; ++i) {
    if (!std::isfinite(v[i])) throw InvalidArgument("vector contains NaN or infinity");
    ss += double(v[i]) * v[i];
  }
  if (cfg_.metric == Metric::Cosine && ss == 0) throw InvalidArgument(for_query ? "zero-norm query under cosine metric" : "zero-norm vector under cosine metric");
  if (!std::isfinite(float(ss))) throw InvalidArgument("vector norm overflows float range");
}

void Collection::prepare(const float* v, std::vector<float>& out) const {
  out.assign(v, v + cfg_.dim);
  if (cfg_.metric == Metric::Cosine) {
    const float n = norm(out.data(), cfg_.dim);
    for (auto& x : out) x /= n;
  }
}

bool Collection::matches(uint32_t iid, const Filter& f) const {
  for (auto& kv : store_->meta(iid)) if (kv.first == f.key && kv.second == f.value) return true;
  return false;
}

void Collection::tombstone(uint32_t iid) {
  store_->mark_deleted(iid);
  for (auto& kv : store_->meta(iid)) {
    auto it = postings_.find(pkey(kv.first, kv.second));
    if (it != postings_.end() && it->second.live > 0) --it->second.live;
  }
}

void Collection::apply_upsert(uint64_t id, const float* raw, const Metadata& md) {
  std::vector<float> v; prepare(raw, v);
  auto it = id_map_.find(id);
  if (it != id_map_.end()) tombstone(it->second);
  const uint32_t iid = store_->add(id, v.data(), md);
  hnsw_->insert(iid);
  id_map_[id] = iid;
  for (auto& kv : md) { auto& p = postings_[pkey(kv.first, kv.second)]; p.ids.push_back(iid); ++p.live; }
}

bool Collection::apply_erase(uint64_t id) {
  auto it = id_map_.find(id);
  if (it == id_map_.end()) return false;
  tombstone(it->second);
  id_map_.erase(it);
  return true;
}

void Collection::rebuild_aux() {
  id_map_.clear(); postings_.clear();
  for (uint32_t i = 0; i < store_->size(); ++i) {
    if (store_->deleted(i)) continue;
    id_map_[store_->label(i)] = i;
    for (auto& kv : store_->meta(i)) { auto& p = postings_[pkey(kv.first, kv.second)]; p.ids.push_back(i); ++p.live; }
  }
}

// ---------- public mutation API ----------

static void write_meta(Writer& w, const Metadata& md) {
  w.put<uint32_t>(uint32_t(md.size()));
  for (auto& kv : md) { w.put_str(kv.first); w.put_str(kv.second); }
}
static Metadata read_meta(Reader& r) {
  uint32_t n = r.get<uint32_t>();
  if (n > r.remaining()) throw CorruptionError("metadata count exceeds buffer");
  Metadata md; md.reserve(n);
  for (uint32_t i = 0; i < n; ++i) { std::string k = r.get_str(); std::string v = r.get_str(); md.emplace_back(std::move(k), std::move(v)); }
  return md;
}

void Collection::upsert(uint64_t id, const float* v, size_t dim, const Metadata& md) {
  validate(v, dim, false);
  for (auto& kv : md) if (kv.first.empty()) throw InvalidArgument("empty metadata key");
  uint64_t lsn = 0;
  {
    std::unique_lock<RWLock> lk(mu_);
    if (persistent_) {
      Writer w; w.put<uint64_t>(id); write_meta(w, md); w.put_bytes(v, dim * sizeof(float));
      lsn = last_lsn_ + 1;
      wal_.append(lsn, OP_UPSERT, w.buf, popts_.sync, popts_.sync_every);  // may throw: nothing applied yet
      last_lsn_ = lsn;
    }
    apply_upsert(id, v, md);
    maybe_auto_compact();
  }
  // Group commit: the durable wait happens outside the exclusive lock so that other writers can append
  // and readers can proceed while this thread (or a peer acting as leader) waits for fdatasync.
  if (persistent_ && popts_.sync == SyncMode::Group) wal_.wait_durable(lsn);
}

bool Collection::erase(uint64_t id) {
  uint64_t lsn = 0; bool removed;
  {
    std::unique_lock<RWLock> lk(mu_);
    if (!id_map_.count(id)) return false;
    if (persistent_) {
      Writer w; w.put<uint64_t>(id);
      lsn = last_lsn_ + 1;
      wal_.append(lsn, OP_DELETE, w.buf, popts_.sync, popts_.sync_every);
      last_lsn_ = lsn;
    }
    removed = apply_erase(id);
    maybe_auto_compact();
  }
  if (persistent_ && popts_.sync == SyncMode::Group) wal_.wait_durable(lsn);
  return removed;
}

// ---------- queries ----------

std::vector<SearchResult> Collection::search(const float* q, size_t dim, const SearchParams& sp, SearchStats* st) const {
  validate(q, dim, true);
  if (sp.k == 0) throw InvalidArgument("k must be >= 1");
  if (sp.k > 100000) throw InvalidArgument("k too large (max 100000)");
  if (sp.ef == 0) throw InvalidArgument("ef must be >= 1");
  std::vector<float> qv; prepare(q, qv);
  std::shared_lock<RWLock> lk(mu_);
  std::vector<SearchResult> res;
  if (store_->live() == 0) return res;
  const DistFn fn = dist_fn(cfg_.metric);
  const size_t d = cfg_.dim;
  SearchStats local; SearchStats* S = st ? st : &local;
  auto live = [&](uint32_t i) { return !store_->deleted(i); };
  const bool filtered = sp.filter.active();

  // Exhaustive scan over `ids` (or all nodes when ids==nullptr).
  auto scan = [&](const std::vector<uint32_t>* ids) {
    std::priority_queue<std::pair<float, uint32_t>> top;
    auto consider = [&](uint32_t i) {
      if (store_->deleted(i)) return;
      if (filtered && !matches(i, sp.filter)) return;
      ++S->dist_computations;
      float dd = fn(qv.data(), store_->vec(i), d);
      if (top.size() < sp.k) top.emplace(dd, i);
      else if (dd < top.top().first) { top.pop(); top.emplace(dd, i); }
    };
    if (ids) for (uint32_t i : *ids) consider(i); else for (uint32_t i = 0; i < store_->size(); ++i) consider(i);
    res.resize(top.size());
    for (size_t i = res.size(); i-- > 0;) { res[i] = {store_->label(top.top().second), top.top().first}; top.pop(); }
  };

  if (sp.exact) { S->strategy_used = 0; scan(nullptr); return res; }

  std::vector<Hnsw::DI> out;
  if (!filtered) {
    S->strategy_used = 1;
    hnsw_->search(qv.data(), sp.k, sp.ef, live, out, S);
  } else {
    const std::vector<uint32_t>* plist = nullptr;
    size_t matching = 0;
    auto pit = postings_.find(pkey(sp.filter.key, sp.filter.value));
    if (pit != postings_.end()) { plist = &pit->second.ids; matching = pit->second.live; }
    if (matching == 0) return res;
    FilterStrategy s = sp.strategy;
    if (s == FilterStrategy::Auto) {
      if (matching <= sp.brute_force_threshold) s = FilterStrategy::BruteForce;                                   // few matches: exact scan of the posting list
      else if (double(matching) >= sp.post_filter_fraction * double(store_->live())) s = FilterStrategy::PostFilter;  // permissive: cheapest
      else s = FilterStrategy::Traversal;                                                                         // selective but too many to scan
    }
    if (s == FilterStrategy::BruteForce) { S->strategy_used = 4; scan(plist); return res; }
    if (s == FilterStrategy::Traversal) {
      S->strategy_used = 2;
      auto acc = [&](uint32_t i) { return !store_->deleted(i) && matches(i, sp.filter); };
      hnsw_->search(qv.data(), sp.k, sp.ef, acc, out, S);
    } else {  // PostFilter: over-fetch, filter, widen until k hits or exhausted
      S->strategy_used = 3;
      size_t ef = std::max(sp.ef, sp.k);
      std::vector<Hnsw::DI> raw;
      for (;;) {
        hnsw_->search(qv.data(), ef, ef, live, raw, S);
        out.clear();
        for (auto& r : raw) if (matches(r.second, sp.filter)) out.push_back(r);
        if (out.size() >= sp.k || ef >= store_->live()) break;
        ef = std::min(ef * 2, store_->live());
      }
      if (out.size() > sp.k) out.resize(sp.k);
    }
  }
  res.reserve(out.size());
  for (auto& r : out) res.push_back({store_->label(r.second), r.first});
  return res;
}

bool Collection::get(uint64_t id, std::vector<float>* out, Metadata* md) const {
  std::shared_lock<RWLock> lk(mu_);
  auto it = id_map_.find(id);
  if (it == id_map_.end()) return false;
  if (out) out->assign(store_->vec(it->second), store_->vec(it->second) + cfg_.dim);
  if (md) *md = store_->meta(it->second);
  return true;
}
bool Collection::contains(uint64_t id) const { std::shared_lock<RWLock> lk(mu_); return id_map_.count(id) != 0; }
size_t Collection::size() const { std::shared_lock<RWLock> lk(mu_); return store_->live(); }

CollectionStats Collection::stats() const {
  std::shared_lock<RWLock> lk(mu_);
  CollectionStats s;
  s.dim = cfg_.dim; s.live = store_->live(); s.tombstones = store_->size() - store_->live();
  s.vector_bytes = store_->memory_bytes(); s.graph_bytes = hnsw_->memory_bytes();
  for (auto& kv : postings_) s.index_bytes += kv.first.capacity() + kv.second.ids.capacity() * 4 + 64;
  s.last_lsn = last_lsn_; s.wal_bytes = persistent_ ? wal_.bytes() : 0; s.max_level = hnsw_->max_level();
  return s;
}

void Collection::compact() { std::unique_lock<RWLock> lk(mu_); compact_locked(); }
size_t Collection::compactions() const { std::shared_lock<RWLock> lk(mu_); return compactions_; }

void Collection::maybe_auto_compact() {
  if (cfg_.auto_compact_ratio <= 0) return;
  const size_t tomb = store_->size() - store_->live();
  if (tomb >= cfg_.auto_compact_min && double(tomb) >= cfg_.auto_compact_ratio * double(store_->size())) compact_locked();
}

void Collection::compact_locked() {
  auto ns = std::make_unique<VectorStore>(cfg_.dim);
  ns->reserve(store_->live());
  for (uint32_t i = 0; i < store_->size(); ++i)
    if (!store_->deleted(i)) ns->add(store_->label(i), store_->vec(i), store_->meta(i));
  auto nh = std::make_unique<Hnsw>(ns.get(), cfg_.metric, cfg_.hnsw);
  for (uint32_t i = 0; i < ns->size(); ++i) nh->insert(i);
  hnsw_ = std::move(nh); store_ = std::move(ns);   // hnsw_ referenced old store_ only until this line
  rebuild_aux();
  ++compactions_;
}

bool Collection::check_invariants(std::string* why) const {
  std::shared_lock<RWLock> lk(mu_);
  auto fail = [&](const char* m) { if (why) *why = m; return false; };
  if (hnsw_->size() != store_->size()) return fail("hnsw size != store size");
  std::string w2;
  if (!hnsw_->check_invariants(&w2)) { if (why) *why = "hnsw: " + w2; return false; }
  size_t live = 0;
  for (uint32_t i = 0; i < store_->size(); ++i) {
    if (store_->deleted(i)) continue;
    ++live;
    auto it = id_map_.find(store_->label(i));
    if (it == id_map_.end() || it->second != i) return fail("live vector missing from id map");
  }
  if (live != store_->live() || live != id_map_.size()) return fail("live count mismatch");
  for (auto& kv : postings_) {
    size_t c = 0;
    for (uint32_t i : kv.second.ids) if (!store_->deleted(i)) ++c;
    if (c != kv.second.live) return fail("posting live count mismatch");
  }
  return true;
}

// ---------- persistence ----------

std::string Collection::serialize_snapshot(uint64_t lsn) const {
  Writer w;
  w.put_bytes(kSnapMagic, 8); w.put<uint32_t>(kSnapVersion); w.put<uint64_t>(lsn);
  w.put<uint32_t>(uint32_t(cfg_.dim)); w.put<uint8_t>(uint8_t(cfg_.metric));
  w.put<uint32_t>(cfg_.hnsw.M); w.put<uint32_t>(cfg_.hnsw.ef_construction); w.put<uint64_t>(cfg_.hnsw.seed);
  w.put<uint8_t>(cfg_.hnsw.heuristic ? 1 : 0);
  w.put<uint64_t>(store_->size());
  for (uint32_t i = 0; i < store_->size(); ++i) {
    w.put<uint64_t>(store_->label(i)); w.put<uint8_t>(store_->deleted(i) ? 1 : 0);
    w.put_bytes(store_->vec(i), cfg_.dim * sizeof(float));
    write_meta(w, store_->meta(i));
  }
  hnsw_->serialize(w);
  w.put<uint32_t>(crc32(w.buf.data(), w.buf.size()));
  return std::move(w.buf);
}

void Collection::load_snapshot(const std::string& data) {
  if (data.size() < 8 + 4 + 8 + 4) throw CorruptionError("snapshot: too small");
  uint32_t stored; std::memcpy(&stored, data.data() + data.size() - 4, 4);
  if (crc32(data.data(), data.size() - 4) != stored) throw CorruptionError("snapshot: checksum mismatch");
  Reader r(data.data(), data.size() - 4);
  char magic[8]; r.get_bytes(magic, 8);
  if (std::memcmp(magic, kSnapMagic, 8) != 0) throw CorruptionError("snapshot: bad magic");
  const uint32_t ver = r.get<uint32_t>();
  if (ver != kSnapVersion) throw Error("snapshot: unsupported format version " + std::to_string(ver));
  last_lsn_ = r.get<uint64_t>();
  const uint32_t dim = r.get<uint32_t>(); const uint8_t metric = r.get<uint8_t>();
  if (dim != cfg_.dim || metric != uint8_t(cfg_.metric)) throw Error("snapshot incompatible with requested dim/metric");
  if (metric > 2) throw CorruptionError("snapshot: bad metric");
  HnswParams hp; hp.M = r.get<uint32_t>(); hp.ef_construction = r.get<uint32_t>(); hp.seed = r.get<uint64_t>();
  hp.heuristic = r.get<uint8_t>() != 0;
  cfg_.hnsw = hp;
  const uint64_t n = r.get<uint64_t>();
  if (n > r.remaining() / (8 + 1 + dim * 4 + 4)) throw CorruptionError("snapshot: record count exceeds data");
  store_ = std::make_unique<VectorStore>(dim);
  store_->reserve(size_t(n));
  std::vector<float> tmp(dim);
  std::vector<uint8_t> dels; dels.reserve(size_t(n));
  for (uint64_t i = 0; i < n; ++i) {
    uint64_t label = r.get<uint64_t>(); uint8_t del = r.get<uint8_t>();
    r.get_bytes(tmp.data(), dim * sizeof(float));
    for (float x : tmp) if (!std::isfinite(x)) throw CorruptionError("snapshot: non-finite vector value");
    Metadata md = read_meta(r);
    store_->add(label, tmp.data(), std::move(md));
    dels.push_back(del);
  }
  for (uint64_t i = 0; i < n; ++i) if (dels[i]) store_->mark_deleted(uint32_t(i));
  hnsw_ = std::make_unique<Hnsw>(store_.get(), cfg_.metric, cfg_.hnsw);
  hnsw_->load(r);
  if (r.remaining() != 0) throw CorruptionError("snapshot: trailing bytes");
  rebuild_aux();
}

std::unique_ptr<Collection> Collection::open(const std::string& dir, const CollectionConfig& cfg, PersistOptions opts, RecoveryInfo* info) {
  namespace fs = std::filesystem;
  fs::create_directories(dir);
  auto c = std::make_unique<Collection>(cfg);
  c->persistent_ = true; c->dir_ = dir; c->popts_ = opts;
  c->lock_ = std::make_unique<DirLock>(dir);
  RecoveryInfo ri;
  const std::string snap = dir + "/snapshot.vf", walp = dir + "/wal.log";
  if (file_exists(snap)) {
    ri.recovered = true;
    auto t0 = Clock::now();
    c->load_snapshot(read_file(snap));
    ri.snapshot_load_ms = ms_since(t0);
    ri.snapshot_lsn = c->last_lsn_;
    if (!file_exists(walp)) throw CorruptionError("wal.log missing for existing collection");
    auto t1 = Clock::now();
    Wal::ReadResult rr = Wal::read_all(walp);
    if (rr.base_lsn > ri.snapshot_lsn) throw CorruptionError("wal base LSN is ahead of snapshot: snapshot lost?");
    for (auto& rec : rr.records) {
      if (rec.lsn <= ri.snapshot_lsn) { ++ri.wal_records_skipped; continue; }
      if (rec.lsn != c->last_lsn_ + 1) throw CorruptionError("wal: gap in LSN sequence");
      Reader r(rec.body.data(), rec.body.size());
      if (rec.op == OP_UPSERT) {
        uint64_t id = r.get<uint64_t>(); Metadata md = read_meta(r);
        std::vector<float> v(cfg.dim);
        if (r.remaining() != cfg.dim * sizeof(float)) throw CorruptionError("wal: bad upsert payload size");
        r.get_bytes(v.data(), cfg.dim * sizeof(float));
        c->apply_upsert(id, v.data(), md);
      } else if (rec.op == OP_DELETE) {
        c->apply_erase(r.get<uint64_t>());
      } else throw CorruptionError("wal: unknown op");
      c->last_lsn_ = rec.lsn; ++ri.wal_records_replayed;
    }
    ri.wal_replay_ms = ms_since(t1);
    ri.wal_bytes_truncated = rr.file_bytes - rr.valid_bytes;
    c->wal_.open_append(walp, rr.valid_bytes);
  } else {
    if (file_exists(walp)) throw CorruptionError("wal.log exists but snapshot.vf is missing");
    atomic_write_file(dir, "snapshot.vf", c->serialize_snapshot(0));
    Wal w; w.create(dir + "/wal.log.tmp", 0); w.close();
    if (::rename((dir + "/wal.log.tmp").c_str(), walp.c_str()) != 0) throw IoError("rename wal failed");
    fsync_dir(dir);
    c->wal_.open_append(walp, 16);
  }
  if (info) *info = ri;
  return c;
}

void Collection::checkpoint() {
  if (!persistent_) throw Error("checkpoint(): collection is not persistent");
  std::lock_guard<std::mutex> cg(ckpt_mu_);
  // Shared lock: blocks writers (so snapshot+LSN are consistent) but lets searches continue.
  std::shared_lock<RWLock> lk(mu_);
  const uint64_t lsn = last_lsn_;
  wal_.sync();
  atomic_write_file(dir_, "snapshot.vf", serialize_snapshot(lsn));  // 1. publish snapshot atomically
  Wal fresh; fresh.create(dir_ + "/wal.log.tmp", lsn);              // 2. build the replacement log
  fresh.close();
  if (::rename((dir_ + "/wal.log.tmp").c_str(), (dir_ + "/wal.log").c_str()) != 0) throw IoError("rename wal failed");
  fsync_dir(dir_);                                                  // 3. publish it
  wal_.open_append(dir_ + "/wal.log", 16);
  // Crash between 1 and 3 is safe: replay skips records with lsn <= snapshot lsn.
}

void Collection::sync() { if (persistent_) { std::unique_lock<RWLock> lk(mu_); wal_.sync(); } }

}  // namespace vf
