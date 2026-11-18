# Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
"""MetalPassthrough for Python: Apple-GPU (Metal) compute from inside a Linux container, zero-copy.

Arrays created with :func:`empty`, :func:`zeros` or :func:`asarray` live in MetalPassthrough arena
memory: 2 MiB guest hugepages that the macOS host wraps as a Metal buffer. NumPy, DLPack and
PyTorch views of them alias the exact pages the GPU reads and writes — nothing is copied.

    import numpy as np, metalpassthrough as mpt

    @mpt.metal_kernel
    def saxpy():
        '''
        kernel void saxpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant float& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            y[i] = fma(a, x[i], y[i]);
        }
        '''

    x = mpt.asarray(np.arange(1 << 20, dtype=np.float32)); y = mpt.zeros_like(x)
    saxpy(x, y, np.float32(2.0), grid=x.size)
    mpt.synchronize(); print(y[:4])
"""

from __future__ import annotations

import ctypes
import inspect
import os
import textwrap
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "device", "stats", "empty", "zeros", "zeros_like", "empty_like", "asarray", "is_device_array",
    "device_address", "synchronize", "flush", "Kernel", "PTXKernel", "metal_kernel", "ptx_kernel",
    "unified_addressing", "ping", "memory_info", "MetalPassthroughError",
]
__version__ = "0.1.0"

# --------------------------------------------------------------------------- C ABI

class MetalPassthroughError(RuntimeError):
    pass


def _load():
    names = [os.environ.get("METALPASSTHROUGH_LIB", ""), "/usr/local/metalpassthrough/lib/libmetalpassthrough.so.1", "libmetalpassthrough.so.1"]
    errs = []
    for n in names:
        if not n:
            continue
        try:
            return ctypes.CDLL(n)
        except OSError as e:
            errs.append(str(e))
    raise MetalPassthroughError("libmetalpassthrough.so.1 not found: " + "; ".join(errs))


_lib = _load()
_c = ctypes
_u32, _u64, _i32, _vp, _cp, _sz = _c.c_uint32, _c.c_uint64, _c.c_int32, _c.c_void_p, _c.c_char_p, _c.c_size_t


class DeviceInfo(_c.Structure):
    _fields_ = [
        ("magic", _u32), ("gpu_cores", _u32), ("name", _c.c_char * 64), ("metal_family", _c.c_char * 32),
        ("host_os", _c.c_char * 48), ("unified_memory_bytes", _u64), ("recommended_working_set", _u64),
        ("max_buffer_length", _u64), ("current_allocated", _u64), ("max_threads_per_tg", _u32),
        ("threadgroup_memory", _u32), ("simd_width", _u32), ("has_unified_memory", _u32),
        ("measured_bandwidth_gbs", _c.c_double), ("host_page_size", _u32), ("abi_version", _u32),
        ("daemon_version", _c.c_char * 16),
    ]

    def __repr__(self):
        return (f"<{self.name.decode()} | {self.gpu_cores}-core GPU | {self.metal_family.decode()} | "
                f"{self.unified_memory_bytes / 2**30:.0f} GiB unified | {self.measured_bandwidth_gbs:.0f} GB/s>")


MAX_PROCS = 64


class ProcInfo(_c.Structure):
    _fields_ = [("pid", _u32), ("session", _u32), ("arena_bytes", _u64), ("dispatches", _u64), ("gpu_ns", _u64),
                ("comm", _c.c_char * 16)]


class Stats(_c.Structure):
    # Must mirror mpt_stats in metalpassthrough_protocol.h exactly (checked against the C library at import).
    _fields_ = [
        ("commands", _u64), ("dispatches", _u64), ("command_buffers", _u64), ("pipeline_compiles", _u64),
        ("pipeline_cache_hits", _u64), ("last_kernel_gpu_ns", _u64), ("total_gpu_ns", _u64), ("arena_bytes", _u64),
        ("arenas", _u32), ("active_sessions_vm", _u32), ("active_containers", _u32), ("doorbells", _u32),
        ("pipelines_resident", _u64), ("vm_arena_bytes", _u64), ("vm_dispatches", _u64), ("vm_total_gpu_ns", _u64),
        ("vm_last_kernel_gpu_ns", _u64), ("nprocs", _u32), ("_pad", _u32), ("procs", ProcInfo * MAX_PROCS),
    ]

    @property
    def processes(self):
        return [self.procs[i] for i in range(min(self.nprocs, MAX_PROCS))]


class _Pipeline(_c.Structure):
    _fields_ = [("handle", _u64), ("max_threads_per_tg", _u32), ("simd_width", _u32), ("static_tg_memory", _u32)]


class _Arg(_c.Structure):
    _fields_ = [("kind", _u32), ("index", _u32), ("ptr", _vp), ("len", _u64)]


class _KernelInfo(_c.Structure):
    _fields_ = [("nparams", _u32), ("params_bytes", _u32), ("shared_bytes", _u32), ("flags", _u32),
                ("param_offset", _u32 * 64), ("param_size", _u32 * 64)]


_sigs = {
    "mpt_default": ([], _vp), "mpt_last_error": ([], _cp), "mpt_strerror": ([_i32], _cp),
    "mpt_device": ([_vp], _c.POINTER(DeviceInfo)), "mpt_get_stats": ([_vp, _c.POINTER(Stats)], _i32),
    "mpt_ping": ([_vp, _c.POINTER(_u64)], _i32), "mpt_unified_addressing": ([_vp], _i32),
    "mpt_alloc": ([_vp, _sz, _sz], _vp), "mpt_free": ([_vp, _vp], None), "mpt_devaddr": ([_vp, _vp], _u64),
    "mpt_memory_info": ([_vp, _c.POINTER(_u64), _c.POINTER(_u64)], _i32),
    "mpt_compile_msl": ([_vp, _cp, _cp, _u32, _c.POINTER(_Pipeline), _cp, _sz], _i32),
    "mpt_compile_ptx": ([_vp, _cp, _cp, _u32, _c.POINTER(_Pipeline), _c.POINTER(_KernelInfo), _cp, _sz], _i32),
    "mpt_dispatch": ([_vp, _c.POINTER(_Pipeline), _c.POINTER(_u32), _c.POINTER(_u32), _u32, _c.POINTER(_Arg), _u32, _u32], _i32),
    "mpt_dispatch_raw": ([_vp, _c.POINTER(_Pipeline), _c.POINTER(_u32), _c.POINTER(_u32), _vp, _u32, _u32, _u32], _i32),
    "mpt_sync": ([_vp, _c.POINTER(_u64)], _i32), "mpt_flush": ([_vp], _i32),
    "mpt_abi_sizeof": ([_cp], _sz),
}
for _n, (_a, _r) in _sigs.items():
    _f = getattr(_lib, _n)
    _f.argtypes, _f.restype = _a, _r

# Guard against Python/C struct drift (a mismatch would corrupt memory, not just misreport).
for _name, _cls in (("mpt_device_info", DeviceInfo), ("mpt_stats", Stats), ("mpt_kernel_info", _KernelInfo)):
    _want = _lib.mpt_abi_sizeof(_name.encode())
    if _want != _c.sizeof(_cls):
        raise MetalPassthroughError(f"ABI mismatch: {_name} is {_want} bytes in libmetalpassthrough, {_c.sizeof(_cls)} in Python")

BIND_BUFFER, BIND_BYTES, BIND_TGMEM = 0, 1, 2
DISPATCH_THREADGROUPS, DISPATCH_THREADS = 0, 1
CF_FAST_MATH, CF_FP64_LOWER, CF_DUMP_MSL = 1, 2, 4
KF_USES_SHARED = 1

_ctx = None


def _session():
    global _ctx
    if _ctx is None:
        _ctx = _lib.mpt_default()
        if not _ctx:
            raise MetalPassthroughError(_lib.mpt_last_error().decode())
    return _ctx


def _check(status: int, what: str):
    if status != 0:
        raise MetalPassthroughError(f"{what}: {_lib.mpt_strerror(status).decode()} ({_lib.mpt_last_error().decode()})")


# --------------------------------------------------------------------------- device / runtime

def device() -> DeviceInfo:
    """Host Apple GPU description (name, cores, Metal family, memory, measured bandwidth)."""
    return _lib.mpt_device(_session()).contents


def stats() -> Stats:
    s = Stats()
    _check(_lib.mpt_get_stats(_session(), _c.byref(s)), "stats")
    return s


def ping() -> float:
    """Round trip through the shared-memory ring, in microseconds."""
    ns = _u64()
    _check(_lib.mpt_ping(_session(), _c.byref(ns)), "ping")
    return ns.value / 1e3


def unified_addressing() -> bool:
    """True if array pointers equal their GPU device addresses (CUDA-style UVA)."""
    return bool(_lib.mpt_unified_addressing(_session()))


def memory_info() -> Tuple[int, int]:
    f, t = _u64(), _u64()
    _lib.mpt_memory_info(_session(), _c.byref(f), _c.byref(t))
    return f.value, t.value


def synchronize() -> float:
    """Waits for all submitted GPU work. Returns GPU execution time (s) since the previous sync."""
    ns = _u64()
    _check(_lib.mpt_sync(_session(), _c.byref(ns)), "synchronize")
    return ns.value / 1e9


def flush() -> None:
    _check(_lib.mpt_flush(_session()), "flush")


# --------------------------------------------------------------------------- zero-copy arrays

class _ArenaOwner:
    """Owns one arena allocation; freed when the last NumPy view is garbage collected."""

    __slots__ = ("ptr", "nbytes", "__weakref__")

    def __init__(self, nbytes: int, align: int = 256):
        ctx = _session()
        self.nbytes = max(int(nbytes), 1)
        self.ptr = _lib.mpt_alloc(ctx, self.nbytes, align)
        if not self.ptr:
            raise MemoryError(f"MetalPassthrough arena allocation of {nbytes} bytes failed: {_lib.mpt_last_error().decode()}")

    def __del__(self):
        if self.ptr and _ctx:
            try:
                # The GPU may still be using it: drain before recycling (matches cudaFree semantics).
                _lib.mpt_sync(_ctx, None)
                _lib.mpt_free(_ctx, self.ptr)
            except Exception:
                pass
            self.ptr = None


def empty(shape, dtype=np.float32, align: int = 256) -> np.ndarray:
    """A NumPy array backed by GPU-visible unified memory (no copy is ever made)."""
    dtype = np.dtype(dtype)
    shape = (shape,) if np.isscalar(shape) else tuple(shape)
    n = int(np.prod(shape)) * dtype.itemsize
    owner = _ArenaOwner(n, align)
    raw = (_c.c_char * owner.nbytes).from_address(owner.ptr)
    arr = np.frombuffer(raw, dtype=np.uint8, count=n).view(dtype).reshape(shape)
    # Keep the owner alive as long as any view exists (ctypes buffer -> owner).
    raw._metalpassthrough_owner = owner  # type: ignore[attr-defined]
    return arr


def zeros(shape, dtype=np.float32) -> np.ndarray:
    a = empty(shape, dtype)
    a.view(np.uint8)[...] = 0
    return a


def empty_like(a: np.ndarray) -> np.ndarray:
    return empty(a.shape, a.dtype)


def zeros_like(a: np.ndarray) -> np.ndarray:
    return zeros(a.shape, a.dtype)


def asarray(a, dtype=None) -> np.ndarray:
    """Returns `a` itself if it already lives in arena memory, else a contiguous arena copy."""
    a = np.asarray(a, dtype=dtype)
    if is_device_array(a) and a.flags.c_contiguous:
        return a
    out = empty(a.shape, a.dtype)
    out[...] = a
    return out


def device_address(a: np.ndarray) -> int:
    """64-bit GPU device address of an arena array (0 if it is ordinary host memory)."""
    return int(_lib.mpt_devaddr(_session(), _c.c_void_p(a.__array_interface__["data"][0])))


def is_device_array(a) -> bool:
    return isinstance(a, np.ndarray) and a.size > 0 and device_address(a) != 0


def to_torch(a: np.ndarray):
    """Zero-copy torch.Tensor view (shares the arena pages; CPU tensor sharing GPU memory)."""
    import torch  # noqa: PLC0415

    try:
        return torch.from_dlpack(a)
    except (RuntimeError, TypeError, AttributeError):  # NumPy < 1.23 has no usable __dlpack__
        return torch.from_numpy(a)


# --------------------------------------------------------------------------- kernels

def _scalar_bytes(v) -> bytes:
    if isinstance(v, np.generic):
        return v.tobytes()
    if isinstance(v, bool):
        return np.uint32(v).tobytes()
    if isinstance(v, int):
        return np.int32(v).tobytes()
    if isinstance(v, float):
        return np.float32(v).tobytes()
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, np.ndarray) and not is_device_array(v):
        return v.tobytes()  # small host arrays / structs become constant bytes
    raise TypeError(f"unsupported kernel argument {type(v).__name__}")


def _grid3(g) -> Tuple[int, int, int]:
    if g is None:
        return (1, 1, 1)
    if np.isscalar(g):
        return (int(g), 1, 1)
    g = tuple(int(x) for x in g)
    return (g + (1, 1, 1))[:3]


class Kernel:
    """A compiled Metal compute pipeline. Positional arguments bind to [[buffer(0)]], [[buffer(1)]], ...

    Arena arrays bind as device buffers (zero-copy); scalars / bytes / host arrays bind as constant data.
    Launch either with `grid=` (total threads, dispatchThreads) or `groups=` (threadgroups).
    """

    def __init__(self, source: str, name: str, fast_math: bool = False):
        self.source, self.name = source, name
        self._p = _Pipeline()
        log = _c.create_string_buffer(65536)
        src = source if "#include" in source else "#include <metal_stdlib>\nusing namespace metal;\n" + source
        st = _lib.mpt_compile_msl(_session(), src.encode(), name.encode(), CF_FAST_MATH if fast_math else 0,
                                   _c.byref(self._p), log, len(log))
        if st != 0:
            raise MetalPassthroughError(f"compiling '{name}' failed:\n{log.value.decode(errors='replace')}")

    @property
    def max_threads_per_threadgroup(self) -> int:
        return self._p.max_threads_per_tg

    @property
    def simd_width(self) -> int:
        return self._p.simd_width

    def __call__(self, *args, grid=None, groups=None, threads=None, threadgroup_memory: Sequence[int] = (), sync=False):
        n = len(args) + len(threadgroup_memory)
        cargs = (_Arg * max(n, 1))()
        keep = []
        for i, a in enumerate(args):
            if isinstance(a, np.ndarray) and is_device_array(a):
                if not a.flags.c_contiguous:
                    raise ValueError(f"argument {i}: device arrays must be C-contiguous")
                cargs[i] = _Arg(BIND_BUFFER, i, a.__array_interface__["data"][0], a.nbytes)
            else:
                b = _scalar_bytes(a)
                buf = _c.create_string_buffer(b, len(b))
                keep.append(buf)
                cargs[i] = _Arg(BIND_BYTES, i, _c.cast(buf, _vp), len(b))
        for k, nbytes in enumerate(threadgroup_memory):
            cargs[len(args) + k] = _Arg(BIND_TGMEM, k, None, int(nbytes))
        if threads is None:
            threads = min(256, self._p.max_threads_per_tg) if groups is None else (min(256, self._p.max_threads_per_tg),)
        tg = (_u32 * 3)(*_grid3(threads))
        if groups is not None:
            mode, g = DISPATCH_THREADGROUPS, _grid3(groups)
        else:
            if grid is None:
                raise ValueError("pass grid= (threads) or groups= (threadgroups)")
            mode, g = DISPATCH_THREADS, _grid3(grid)
        _check(_lib.mpt_dispatch(_session(), _c.byref(self._p), (_u32 * 3)(*g), tg, mode, cargs, n, 0), f"dispatch {self.name}")
        if sync:
            return synchronize()
        return None


class PTXKernel:
    """An NVIDIA PTX entry point translated to Metal. Arguments follow the PTX .param list:
    arena arrays pass their device address (pointer params); scalars are packed to the param size."""

    def __init__(self, ptx: str, entry: str, lower_fp64: bool = False):
        self.entry = entry
        self._p, self._info = _Pipeline(), _KernelInfo()
        log = _c.create_string_buffer(1 << 20)
        st = _lib.mpt_compile_ptx(_session(), ptx.encode(), entry.encode(), CF_FP64_LOWER if lower_fp64 else 0,
                                   _c.byref(self._p), _c.byref(self._info), log, len(log))
        if st != 0:
            raise MetalPassthroughError(f"PTX entry '{entry}' failed:\n{log.value.decode(errors='replace')}")
        self.warnings = log.value.decode(errors="replace")

    def __call__(self, *args, grid, block, shared_mem: int = 0, sync=False):
        info = self._info
        if len(args) != info.nparams:
            raise TypeError(f"{self.entry} expects {info.nparams} arguments, got {len(args)}")
        blob = bytearray(max(info.params_bytes, 16))
        for i, a in enumerate(args):
            off, size = info.param_offset[i], info.param_size[i]
            if isinstance(a, np.ndarray):
                if not is_device_array(a):
                    raise ValueError(f"argument {i}: pass MetalPassthrough arrays (mpt.asarray) for pointer parameters")
                b = np.uint64(device_address(a)).tobytes()
            elif isinstance(a, np.generic):
                b = a.tobytes()
            elif isinstance(a, float):
                b = (np.float64(a) if size == 8 else np.float32(a)).tobytes()
            elif isinstance(a, int):
                b = (np.int64(a) if size == 8 else np.int32(a)).tobytes()
            else:
                raise TypeError(f"unsupported PTX argument {type(a).__name__}")
            if len(b) != size:
                raise TypeError(f"argument {i}: {len(b)} bytes given, PTX parameter is {size} bytes")
            blob[off:off + size] = b
        buf = (_c.c_char * len(blob)).from_buffer(blob)
        tgmem = info.shared_bytes + shared_mem if info.flags & KF_USES_SHARED else 0
        _check(_lib.mpt_dispatch_raw(_session(), _c.byref(self._p), (_u32 * 3)(*_grid3(grid)), (_u32 * 3)(*_grid3(block)),
                                      _c.cast(buf, _vp), info.params_bytes, tgmem, 0), f"launch {self.entry}")
        if sync:
            return synchronize()
        return None


def _source_of(fn) -> str:
    src = fn.__doc__ or ""
    try:
        ret = fn()
        if isinstance(ret, str) and ret.strip():
            src = ret
    except TypeError:
        pass
    if not src.strip():
        raise ValueError(f"{fn.__name__}: put the kernel source in the docstring or return it")
    return textwrap.dedent(src)


def metal_kernel(fn=None, *, name: Optional[str] = None, fast_math: bool = False):
    """Decorator: the function's docstring (or return value) is Metal Shading Language source;
    the kernel function named like the Python function (or `name=`) is compiled on first use."""

    def wrap(f):
        k: dict = {}

        def call(*args, **kw):
            if "k" not in k:
                k["k"] = Kernel(_source_of(f), name or f.__name__, fast_math=fast_math)
            return k["k"](*args, **kw)

        call.__name__, call.__doc__, call.__wrapped__ = f.__name__, f.__doc__, f
        return call

    return wrap(fn) if fn is not None else wrap


def ptx_kernel(fn=None, *, entry: Optional[str] = None, lower_fp64: bool = False):
    """Decorator: the function's docstring (or return value) is NVIDIA PTX; calls launch `entry`
    (default: the function name) with CUDA-style grid=/block= on the Apple GPU."""

    def wrap(f):
        k: dict = {}

        def call(*args, **kw):
            if "k" not in k:
                k["k"] = PTXKernel(_source_of(f), entry or f.__name__, lower_fp64=lower_fp64)
            return k["k"](*args, **kw)

        call.__name__, call.__doc__, call.__wrapped__ = f.__name__, f.__doc__, f
        return call

    return wrap(fn) if fn is not None else wrap
