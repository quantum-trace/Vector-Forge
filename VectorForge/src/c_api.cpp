#include "vectorforge/c_api.h"
#include <cstring>
#include <memory>
#include "vectorforge/collection.hpp"

struct vf_collection { std::unique_ptr<vf::Collection> impl; };

namespace {
void set_err(char* err, size_t n, const char* msg) {
  if (err && n) { std::strncpy(err, msg, n - 1); err[n - 1] = 0; }
}
template <class F> int guarded(char* err, size_t n, F&& f) {
  try { return f(); }
  catch (const std::exception& e) { set_err(err, n, e.what()); return -1; }
  catch (...) { set_err(err, n, "unknown error"); return -2; }
}
}  // namespace

extern "C" {

vf_collection* vf_open(const char* dir, uint32_t dim, int metric, uint32_t M, uint32_t efc, int sync, char* err, size_t n) {
  try {
    vf::CollectionConfig cfg; cfg.dim = dim; cfg.metric = static_cast<vf::Metric>(metric);
    if (M) cfg.hnsw.M = M;
    if (efc) cfg.hnsw.ef_construction = efc;
    if (metric < 0 || metric > 2) throw vf::InvalidArgument("bad metric");
    auto h = std::make_unique<vf_collection>();
    if (dir) {
      vf::PersistOptions po; po.sync = sync == 0 ? vf::SyncMode::Always : sync == 1 ? vf::SyncMode::EveryN : sync == 3 ? vf::SyncMode::Group : vf::SyncMode::None;
      h->impl = vf::Collection::open(dir, cfg, po);
    } else h->impl = std::make_unique<vf::Collection>(cfg);
    return h.release();
  } catch (const std::exception& e) { set_err(err, n, e.what()); return nullptr; }
}

void vf_close(vf_collection* c) { delete c; }

int vf_upsert(vf_collection* c, uint64_t id, const float* v, uint32_t dim, const char* const* keys, const char* const* vals,
              uint32_t nmeta, char* err, size_t n) {
  return guarded(err, n, [&] {
    vf::Metadata md;
    for (uint32_t i = 0; i < nmeta; ++i) md.emplace_back(keys[i], vals[i]);
    c->impl->upsert(id, v, dim, md);
    return 0;
  });
}

int vf_delete(vf_collection* c, uint64_t id, char* err, size_t n) {
  return guarded(err, n, [&] { return c->impl->erase(id) ? 1 : 0; });
}

int64_t vf_search(vf_collection* c, const float* q, uint32_t dim, uint32_t k, uint32_t ef, const char* fk, const char* fv,
                  uint64_t* out_ids, float* out_dist, char* err, size_t n) {
  return guarded(err, n, [&]() -> int {
    vf::SearchParams sp; sp.k = k; sp.ef = ef ? ef : 64;
    if (fk && *fk) { sp.filter.key = fk; sp.filter.value = fv ? fv : ""; }
    auto r = c->impl->search(q, dim, sp);
    for (size_t i = 0; i < r.size(); ++i) { out_ids[i] = r[i].id; out_dist[i] = r[i].distance; }
    return int(r.size());
  });
}

int vf_checkpoint(vf_collection* c, char* err, size_t n) { return guarded(err, n, [&] { c->impl->checkpoint(); return 0; }); }
uint64_t vf_size(vf_collection* c) { return c->impl->size(); }

}  // extern "C"
