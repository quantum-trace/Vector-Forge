#pragma once
// Hierarchical Navigable Small World graph (Malkov & Yashunin, TPAMI 2020).
//
// Design notes
//  * Level-0 links live in one flat array (cache-friendly); upper layers are sparse
//    and stored per node.
//  * Layer assignment is a pure function of (internal id, seed). This makes index
//    construction deterministic and lets WAL replay rebuild the *same* graph.
//  * Deletion is tombstone-based: deleted nodes stay navigable but are never returned.
//  * `heuristic=false` selects the "simple" closest-M rule (used for the ablation study).
//  * Not internally synchronized: the Collection provides reader/writer locking.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <vector>
#include "distance.hpp"
#include "io.hpp"
#include "store.hpp"
#include "types.hpp"

namespace vf {

struct HnswParams {
  uint32_t M = 16;                // max links per node on layers >= 1 (layer 0 allows 2*M)
  uint32_t ef_construction = 200; // candidate-list size during insertion
  uint64_t seed = 42;
  bool heuristic = true;          // Algorithm 4 neighbour selection vs. simple closest-M
};

class Hnsw {
 public:
  using DI = std::pair<float, uint32_t>;
  static constexpr uint32_t NONE = 0xFFFFFFFFu;

  Hnsw(const VectorStore* store, Metric metric, HnswParams p)
      : store_(store), p_(p), fn_(dist_fn(metric)), M0_(2 * p.M),
        mult_(1.0 / std::log(double(std::max<uint32_t>(2, p.M)))) {
    if (p.M < 2 || p.M > 256) throw InvalidArgument("hnsw: M must be in [2,256]");
    if (p.ef_construction < p.M) throw InvalidArgument("hnsw: ef_construction must be >= M");
  }

  size_t size() const { return level_.size(); }
  const HnswParams& params() const { return p_; }
  int max_level() const { return max_level_; }
  uint32_t entry() const { return entry_; }

  // Insert node `id` (its vector must already be in the store; ids must be sequential).
  void insert(uint32_t id, SearchStats* st = nullptr) {
    if (id != level_.size()) throw Error("hnsw: non-sequential insert");
    const int lvl = level_for(id);
    level_.push_back(uint8_t(lvl));
    l0_.resize(l0_.size() + (M0_ + 1), 0);
    upper_.emplace_back(size_t(lvl) * (p_.M + 1), 0u);
    if (entry_ == NONE) { entry_ = id; max_level_ = lvl; return; }

    const float* q = store_->vec(id);
    uint32_t ep = entry_;
    float epd = d(q, ep, st);
    for (int l = max_level_; l > lvl; --l) greedy(q, ep, epd, l, st);

    std::vector<DI> W, eps;
    std::vector<uint32_t> chosen;
    auto all = [](uint32_t) { return true; };
    for (int l = std::min(lvl, max_level_); l >= 0; --l) {
      eps.assign(1, DI{epd, ep});
      search_layer(q, eps, p_.ef_construction, l, all, W, st);
      select(W, p_.M, chosen);
      uint32_t* L = links(id, l);
      L[0] = uint32_t(chosen.size());
      for (size_t i = 0; i < chosen.size(); ++i) L[1 + i] = chosen[i];
      bool reachable = false;
      for (uint32_t nb : chosen) reachable |= add_link(nb, id, l, st);
      // Reachability guarantee: neighbour-selection can prune the new node from *every*
      // neighbour list (e.g. many exact duplicates, or a far outlier among dense points),
      // leaving it unreachable. Force one in-link from the closest chosen neighbour.
      if (!reachable && !chosen.empty()) force_link(chosen[0], id, l, st);
      ep = W.front().second; epd = W.front().first;
    }
    if (lvl > max_level_) { entry_ = id; max_level_ = lvl; }
  }

  // k-NN search. `accept(id)` decides whether a node may appear in the results
  // (used for tombstones and metadata filters); rejected nodes are still traversed.
  template <class Accept>
  void search(const float* q, size_t k, size_t ef, Accept&& accept, std::vector<DI>& out,
              SearchStats* st = nullptr, size_t max_visits = SIZE_MAX) const {
    out.clear();
    if (entry_ == NONE || k == 0) return;
    ef = std::max(ef, k);
    uint32_t ep = entry_;
    float epd = d(q, ep, st);
    for (int l = max_level_; l >= 1; --l) greedy(q, ep, epd, l, st);
    std::vector<DI> eps{DI{epd, ep}};
    search_layer(q, eps, ef, 0, accept, out, st, max_visits);
    if (out.size() > k) out.resize(k);
  }

  size_t memory_bytes() const {
    size_t m = l0_.size() * 4 + level_.size();
    for (auto& u : upper_) m += u.size() * 4 + sizeof(u);
    return m;
  }

  void serialize(Writer& w) const {
    w.put<uint32_t>(entry_); w.put<int32_t>(max_level_); w.put<uint64_t>(level_.size());
    for (uint32_t id = 0; id < level_.size(); ++id) {
      w.put<uint8_t>(level_[id]);
      for (int l = 0; l <= level_[id]; ++l) {
        const uint32_t* L = links(id, l);
        w.put<uint32_t>(L[0]);
        w.put_bytes(L + 1, size_t(L[0]) * 4);
      }
    }
  }

  // Validates structure aggressively: a CRC passing is not enough to trust a graph.
  void load(Reader& r) {
    level_.clear(); l0_.clear(); upper_.clear();
    entry_ = r.get<uint32_t>(); max_level_ = r.get<int32_t>();
    const uint64_t n = r.get<uint64_t>();
    if (n != store_->size()) throw CorruptionError("hnsw: node count != vector count");
    if (n == 0 && entry_ != NONE) throw CorruptionError("hnsw: entry on empty graph");
    if (n > 0 && entry_ >= n) throw CorruptionError("hnsw: entry out of range");
    level_.resize(n); l0_.assign(size_t(n) * (M0_ + 1), 0); upper_.resize(n);
    for (uint32_t id = 0; id < n; ++id) {
      uint8_t lv = r.get<uint8_t>();
      if (lv > 63) throw CorruptionError("hnsw: level out of range");
      level_[id] = lv; upper_[id].assign(size_t(lv) * (p_.M + 1), 0u);
      for (int l = 0; l <= lv; ++l) {
        uint32_t cnt = r.get<uint32_t>();
        if (cnt > cap(l)) throw CorruptionError("hnsw: link count exceeds capacity");
        uint32_t* L = links(id, l);
        L[0] = cnt;
        r.get_bytes(L + 1, size_t(cnt) * 4);
        for (uint32_t i = 0; i < cnt; ++i)
          if (L[1 + i] >= n || L[1 + i] == id) throw CorruptionError("hnsw: invalid neighbour id");
      }
    }
    if (n > 0 && (max_level_ < 0 || max_level_ != level_[entry_])) throw CorruptionError("hnsw: max_level mismatch");
    // Neighbours referenced on layer l must themselves exist on layer l.
    for (uint32_t id = 0; id < n; ++id)
      for (int l = 1; l <= level_[id]; ++l) {
        const uint32_t* L = links(id, l);
        for (uint32_t i = 0; i < L[0]; ++i) if (level_[L[1 + i]] < l) throw CorruptionError("hnsw: link to node absent from layer");
      }
  }

  // Structural self-check used by tests: degree bounds, no self-loops, layer membership.
  bool check_invariants(std::string* why = nullptr) const {
    auto fail = [&](const char* m) { if (why) *why = m; return false; };
    for (uint32_t id = 0; id < level_.size(); ++id)
      for (int l = 0; l <= level_[id]; ++l) {
        const uint32_t* L = links(id, l);
        if (L[0] > cap(l)) return fail("degree exceeds capacity");
        for (uint32_t i = 0; i < L[0]; ++i) {
          uint32_t nb = L[1 + i];
          if (nb >= level_.size()) return fail("neighbour out of range");
          if (nb == id) return fail("self loop");
          if (level_[nb] < l) return fail("neighbour not on layer");
        }
      }
    return true;
  }

 private:
  struct Visited {
    std::vector<uint32_t> mark; uint32_t epoch = 0;
    void next(size_t n) {
      if (mark.size() < n) mark.resize(n, 0);
      if (++epoch == 0) { std::fill(mark.begin(), mark.end(), 0u); epoch = 1; }
    }
  };
  static Visited& visited() { static thread_local Visited v; return v; }

  size_t cap(int l) const { return l == 0 ? M0_ : p_.M; }
  uint32_t* links(uint32_t id, int l) {
    return l == 0 ? &l0_[size_t(id) * (M0_ + 1)] : &upper_[id][size_t(l - 1) * (p_.M + 1)];
  }
  const uint32_t* links(uint32_t id, int l) const {
    return l == 0 ? &l0_[size_t(id) * (M0_ + 1)] : &upper_[id][size_t(l - 1) * (p_.M + 1)];
  }
  float d(const float* q, uint32_t id, SearchStats* st) const {
    if (st) ++st->dist_computations;
    return fn_(q, store_->vec(id), store_->dim());
  }

  static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
  }
  int level_for(uint32_t id) const {
    const uint64_t r = splitmix64(uint64_t(id) ^ splitmix64(p_.seed));
    const double u = (double(r >> 11) + 0.5) / 9007199254740992.0;  // (0,1)
    return std::min(int(std::floor(-std::log(u) * mult_)), 32);
  }

  void greedy(const float* q, uint32_t& cur, float& cd, int l, SearchStats* st) const {
    bool changed = true;
    while (changed) {
      changed = false;
      const uint32_t* L = links(cur, l);
      for (uint32_t i = 0; i < L[0]; ++i) {
        const uint32_t nb = L[1 + i];
        const float dn = d(q, nb, st);
        if (dn < cd) { cd = dn; cur = nb; changed = true; }
      }
    }
  }

  template <class Accept>
  void search_layer(const float* q, const std::vector<DI>& eps, size_t ef, int layer, Accept&& accept,
                    std::vector<DI>& out, SearchStats* st, size_t max_visits = SIZE_MAX) const {
    Visited& vl = visited();
    vl.next(level_.size());
    std::priority_queue<DI, std::vector<DI>, std::greater<DI>> cand;  // closest first
    std::priority_queue<DI> top;                                       // worst on top
    for (const DI& e : eps) {
      vl.mark[e.second] = vl.epoch;
      cand.push(e);
      if (accept(e.second)) top.push(e);
    }
    while (top.size() > ef) top.pop();
    size_t visits = 0;
    while (!cand.empty()) {
      const DI c = cand.top();
      if (top.size() >= ef && c.first > top.top().first) break;
      cand.pop();
      const uint32_t* L = links(c.second, layer);
      const uint32_t cnt = L[0];
      for (uint32_t i = 0; i < cnt; ++i) __builtin_prefetch(store_->vec(L[1 + i]));
      for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t nb = L[1 + i];
        if (vl.mark[nb] == vl.epoch) continue;
        vl.mark[nb] = vl.epoch;
        const float dn = d(q, nb, st);
        ++visits;
        if (top.size() < ef || dn < top.top().first) {
          cand.push(DI{dn, nb});
          if (accept(nb)) { top.push(DI{dn, nb}); if (top.size() > ef) top.pop(); }
        }
      }
      if (visits >= max_visits) break;
    }
    out.resize(top.size());
    for (size_t i = out.size(); i-- > 0;) { out[i] = top.top(); top.pop(); }  // ascending
  }

  // `cands` must be sorted ascending by distance to the base point.
  void select(const std::vector<DI>& cands, size_t m, std::vector<uint32_t>& out) const {
    out.clear();
    if (cands.size() <= m) { for (auto& c : cands) out.push_back(c.second); return; }
    if (!p_.heuristic) { for (size_t i = 0; i < m; ++i) out.push_back(cands[i].second); return; }
    for (const DI& c : cands) {
      if (out.size() >= m) break;
      bool good = true;
      for (uint32_t r : out)
        if (fn_(store_->vec(c.second), store_->vec(r), store_->dim()) < c.first) { good = false; break; }
      if (good) out.push_back(c.second);
    }
  }

  // Returns true if `id` is present in nb's list afterwards.
  bool add_link(uint32_t nb, uint32_t id, int l, SearchStats* st) {
    uint32_t* L = links(nb, l);
    const size_t c = cap(l);
    if (L[0] < c) { L[1 + L[0]] = id; ++L[0]; return true; }
    std::vector<DI> cands;
    cands.reserve(c + 1);
    const float* base = store_->vec(nb);
    for (uint32_t i = 0; i < L[0]; ++i) cands.push_back(DI{d(base, L[1 + i], st), L[1 + i]});
    cands.push_back(DI{d(base, id, st), id});
    // Ties (exact duplicates) are broken newest-first so fresh nodes are not systematically dropped.
    std::sort(cands.begin(), cands.end(), [](const DI& a, const DI& b) { return a.first != b.first ? a.first < b.first : a.second > b.second; });
    std::vector<uint32_t> chosen;
    select(cands, c, chosen);
    L[0] = uint32_t(chosen.size());
    bool kept = false;
    for (size_t i = 0; i < chosen.size(); ++i) { L[1 + i] = chosen[i]; kept |= (chosen[i] == id); }
    return kept;
  }

  // Replace nb's farthest link on layer l with `id` (full lists only; otherwise append).
  void force_link(uint32_t nb, uint32_t id, int l, SearchStats* st) {
    uint32_t* L = links(nb, l);
    if (L[0] < cap(l)) { L[1 + L[0]] = id; ++L[0]; return; }
    const float* base = store_->vec(nb);
    uint32_t worst = 0; float wd = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < L[0]; ++i) { float di = d(base, L[1 + i], st); if (di >= wd) { wd = di; worst = i; } }
    L[1 + worst] = id;
  }

  const VectorStore* store_;
  HnswParams p_;
  DistFn fn_;
  size_t M0_;
  double mult_;
  uint32_t entry_ = NONE;
  int max_level_ = -1;
  std::vector<uint8_t> level_;
  std::vector<uint32_t> l0_;
  std::vector<std::vector<uint32_t>> upper_;
};

}  // namespace vf
