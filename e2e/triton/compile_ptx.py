# Compiles Triton kernels ahead of time to NVIDIA PTX (sm_80) on a machine without an NVIDIA GPU.
# Output: e2e/triton/triton_kernels.ptx + .json (entry names and launch metadata).
import json, sys
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

@triton.jit
def softmax_kernel(out_ptr, in_ptr, row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + row * row_stride + cols, mask=cols < n_cols, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    tl.store(out_ptr + row * row_stride + cols, num / tl.sum(num, axis=0), mask=cols < n_cols)

target = GPUTarget("cuda", 80, 32)
out = {}
ptx_all = []
for fn, sig, consts in [
    (add_kernel, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"}, {"BLOCK": 1024}),
    (softmax_kernel, {"out_ptr": "*fp32", "in_ptr": "*fp32", "row_stride": "i32", "n_cols": "i32", "BLOCK": "constexpr"}, {"BLOCK": 1024}),
]:
    src = triton.compiler.ASTSource(fn=fn, signature=sig, constexprs=consts)
    k = triton.compile(src, target=target, options={"num_warps": 4})
    out[fn.__name__] = {"name": k.metadata.name, "num_warps": k.metadata.num_warps, "shared": k.metadata.shared}
    open(f"{sys.argv[1]}/{fn.__name__}.ptx", "w").write(k.asm["ptx"])
json.dump(out, open(f"{sys.argv[1]}/triton_kernels.json", "w"), indent=1)
print(json.dumps(out, indent=1), "triton", triton.__version__)
