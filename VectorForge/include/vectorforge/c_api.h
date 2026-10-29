#ifndef VECTORFORGE_C_API_H
#define VECTORFORGE_C_API_H
/* Stable C ABI used by the Python SDK (ctypes). All functions return 0 on success
 * (or a non-negative count for vf_search) and a negative value on error; the message
 * is copied into err (if non-NULL). No C++ exception crosses this boundary. */
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct vf_collection vf_collection;
/* dir==NULL => in-memory. metric: 0=L2 1=InnerProduct 2=Cosine. sync: 0=Always 1=EveryN 2=None 3=Group(commit) */
vf_collection* vf_open(const char* dir, uint32_t dim, int metric, uint32_t M, uint32_t ef_construction,
                       int sync, char* err, size_t errlen);
void vf_close(vf_collection* c);
int vf_upsert(vf_collection* c, uint64_t id, const float* v, uint32_t dim, const char* const* keys,
              const char* const* vals, uint32_t nmeta, char* err, size_t errlen);
int vf_delete(vf_collection* c, uint64_t id, char* err, size_t errlen); /* returns 1 if existed, 0 if not */
int64_t vf_search(vf_collection* c, const float* q, uint32_t dim, uint32_t k, uint32_t ef, const char* filter_key,
                  const char* filter_val, uint64_t* out_ids, float* out_dist, char* err, size_t errlen);
int vf_checkpoint(vf_collection* c, char* err, size_t errlen);
uint64_t vf_size(vf_collection* c);
#ifdef __cplusplus
}
#endif
#endif
