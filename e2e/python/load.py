# Keeps the GPU busy for a few seconds (used to demo apple-smi's process table).
import sys, time, numpy as np, metalpassthrough as mpt

@mpt.metal_kernel
def burn():
    """
    kernel void burn(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
        float v = x[i];
        for (int k = 0; k < 256; k++) v = fma(v, 0.999f, 0.001f);
        x[i] = v;
    }
    """

x = mpt.zeros(1 << 24, np.float32)
t_end = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 5)
while time.time() < t_end:
    burn(x, grid=x.size)
    mpt.synchronize()
