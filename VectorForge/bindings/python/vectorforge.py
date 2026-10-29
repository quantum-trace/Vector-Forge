"""VectorForge Python SDK (ctypes over the stable C ABI in include/vectorforge/c_api.h).

    import vectorforge as vf
    db = vf.Collection(dim=384, metric="cosine", path="./data/docs")   # path=None => in-memory
    db.upsert(1, vector, {"category": "database"})
    hits = db.search(query_vector, k=5, filter=("category", "database"))   # [(id, distance), ...]
    db.checkpoint(); db.close()

Thread-safe (the engine uses a reader/writer lock). The GIL is released during native calls.
"""
from __future__ import annotations

import ctypes
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Collection", "VectorForgeError"]

_METRICS = {"l2": 0, "ip": 1, "inner_product": 1, "cosine": 2}
_SYNC = {"always": 0, "every_n": 1, "none": 2, "group": 3}
_ERRLEN = 512


class VectorForgeError(RuntimeError):
    pass


def _find_lib() -> str:
    env = os.environ.get("VECTORFORGE_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in ("../../build/libvectorforge.so", "libvectorforge.so", "../../build/libvectorforge.dylib"):
        p = os.path.normpath(os.path.join(here, rel))
        if os.path.exists(p):
            return p
    raise VectorForgeError("libvectorforge not found; run `make` or set VECTORFORGE_LIB")


_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    L = ctypes.CDLL(_find_lib())
    c_float_p = ctypes.POINTER(ctypes.c_float)
    L.vf_open.restype = ctypes.c_void_p
    L.vf_open.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_size_t]
    L.vf_close.argtypes = [ctypes.c_void_p]
    L.vf_upsert.restype = ctypes.c_int
    L.vf_upsert.argtypes = [ctypes.c_void_p, ctypes.c_uint64, c_float_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char_p),
                            ctypes.POINTER(ctypes.c_char_p), ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t]
    L.vf_delete.restype = ctypes.c_int
    L.vf_delete.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t]
    L.vf_search.restype = ctypes.c_int64
    L.vf_search.argtypes = [ctypes.c_void_p, c_float_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p,
                            ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64), c_float_p, ctypes.c_char_p, ctypes.c_size_t]
    L.vf_checkpoint.restype = ctypes.c_int
    L.vf_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    L.vf_size.restype = ctypes.c_uint64
    L.vf_size.argtypes = [ctypes.c_void_p]
    _lib = L
    return L


def _floats(v: Sequence[float], dim: int):
    """Returns (ctypes pointer, keepalive). NumPy float32 C-contiguous arrays are passed zero-copy."""
    if hasattr(v, "ctypes") and getattr(v, "dtype", None) is not None and str(v.dtype) == "float32" and v.flags["C_CONTIGUOUS"]:
        if v.size != dim:
            raise VectorForgeError(f"dimension mismatch: expected {dim}, got {v.size}")
        return v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), v
    if len(v) != dim:
        raise VectorForgeError(f"dimension mismatch: expected {dim}, got {len(v)}")
    arr = (ctypes.c_float * dim)(*[float(x) for x in v])
    return arr, arr


class Collection:
    def __init__(self, dim: int, metric: str = "l2", path: Optional[str] = None, M: int = 16,
                 ef_construction: int = 200, sync: str = "always"):
        if metric not in _METRICS:
            raise VectorForgeError(f"unknown metric {metric!r}")
        if sync not in _SYNC:
            raise VectorForgeError(f"unknown sync mode {sync!r}")
        self._L = _load()
        self.dim = int(dim)
        err = ctypes.create_string_buffer(_ERRLEN)
        h = self._L.vf_open(path.encode() if path else None, self.dim, _METRICS[metric], M, ef_construction, _SYNC[sync], err, _ERRLEN)
        if not h:
            raise VectorForgeError(err.value.decode())
        self._h = ctypes.c_void_p(h)

    def _check(self, rc, err):
        if rc < 0:
            raise VectorForgeError(err.value.decode() or "native error")
        return rc

    def upsert(self, id: int, vector: Sequence[float], metadata: Optional[Dict[str, str]] = None) -> None:
        md = metadata or {}
        keys = (ctypes.c_char_p * len(md))(*[str(k).encode() for k in md])
        vals = (ctypes.c_char_p * len(md))(*[str(v).encode() for v in md.values()])
        err = ctypes.create_string_buffer(_ERRLEN)
        fp, _keep = _floats(vector, self.dim)
        self._check(self._L.vf_upsert(self._h, id, fp, self.dim, keys, vals, len(md), err, _ERRLEN), err)

    def delete(self, id: int) -> bool:
        err = ctypes.create_string_buffer(_ERRLEN)
        return self._check(self._L.vf_delete(self._h, id, err, _ERRLEN), err) == 1

    def search(self, vector: Sequence[float], k: int = 10, ef: int = 64,
               filter: Optional[Tuple[str, str]] = None) -> List[Tuple[int, float]]:
        ids = (ctypes.c_uint64 * k)()
        dist = (ctypes.c_float * k)()
        fk, fv = (filter[0].encode(), str(filter[1]).encode()) if filter else (None, None)
        err = ctypes.create_string_buffer(_ERRLEN)
        fp, _keep = _floats(vector, self.dim)
        n = self._check(self._L.vf_search(self._h, fp, self.dim, k, ef, fk, fv, ids, dist, err, _ERRLEN), err)
        return [(int(ids[i]), float(dist[i])) for i in range(n)]

    def checkpoint(self) -> None:
        err = ctypes.create_string_buffer(_ERRLEN)
        self._check(self._L.vf_checkpoint(self._h, err, _ERRLEN), err)

    def __len__(self) -> int:
        return int(self._L.vf_size(self._h))

    def close(self) -> None:
        if self._h:
            self._L.vf_close(self._h)
            self._h = ctypes.c_void_p(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
