# Phase-2 feasibility probe (guest side). Allocates 2 MiB hugepages, stamps each with a random
# nonce, lets the host locate them in guest RAM, then has the Apple GPU transform them in place.
import ctypes, mmap, os, socket, struct, sys, time
MAP_HUGETLB = 0x40000
HP = 2 << 20
n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
print(open('/proc/meminfo').read().split('HugePages_Total:')[1].split('\n')[0].strip(), "hugepages reserved")
m = mmap.mmap(-1, n * HP, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | MAP_HUGETLB)
nonce = os.urandom(16)
for i in range(n):
    m[i*HP:i*HP+24] = nonce + struct.pack('<Q', i)
s = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
s.connect((2, 2049))
s.sendall(b'CGPB' + nonce + struct.pack('<I', n))
r = s.recv(9); assert r[:5] == b'FOUND', r
print("host located", struct.unpack('<I', r[5:9])[0], "of", n, "hugepages")
cnt = n * HP // 4
arr = (ctypes.c_float * cnt).from_buffer(m)
import array
src = array.array('f', range(1 << 20))  # 4 MiB pattern
b = memoryview(m)
pat = src.tobytes()
for off in range(0, n * HP, len(pat)):
    b[off:off+len(pat)] = pat
t = time.perf_counter()
s.sendall(b'RUN!')
r = s.recv(12); assert r[:4] == b'DONE', r
dt = time.perf_counter() - t
gpu_ns = struct.unpack('<Q', r[4:12])[0]
bad = 0
for idx in list(range(0, cnt, 9973)) + [cnt - 1]:
    want = float(idx % (1 << 20)) * 2 + 1
    if arr[idx] != want: bad += 1; print("MISMATCH", idx, arr[idx], want)
print(f"GPU transformed {n*2} MiB of guest memory in place: gpu={gpu_ns/1e3:.1f} us round-trip={dt*1e3:.2f} ms mismatches={bad}")
sys.exit(1 if bad else 0)
