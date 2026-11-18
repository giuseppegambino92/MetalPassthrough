// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
// Regression tests for SIMT reconvergence: divergent lanes must re-join before bar.sync.
import MetalPassthroughHost
import XCTest
final class DivergenceTests: XCTestCase {
    func testSgemmShapes() throws {
        let k = try PTXToMSLTranslator.translate(ptx: PTXTranslatorTests.ptx, entry: "tiled_sgemm")
        for (M, N, K) in [(32, 32, 32), (32, 33, 32), (32, 32, 33), (33, 32, 32), (16, 17, 16), (1, 17, 16)] {
            var s: UInt64 = 1
            let a: [Float] = (0..<M * K).map { _ in s = s &* 6364136223846793005 &+ 1; return Float(s >> 60) }
            let b: [Float] = (0..<K * N).map { _ in s = s &* 6364136223846793005 &+ 1; return Float(s >> 60) }
            let r = PTXRunner()
            let A = r.buffer(a), B = r.buffer(b), C = r.zeros(M * N * 4)
            try r.launch(k, grid: ((N + 15) / 16, (M + 15) / 16, 1), block: (16, 16, 1), [.ptr(A), .ptr(B), .ptr(C), .i32(Int32(M)), .i32(Int32(N)), .i32(Int32(K))])
            let c = r.read(C, M * N, as: Float.self)
            var bad = [(Int, Int, Float, Float)]()
            for i in 0..<M { for j in 0..<N { var acc: Float = 0; for kk in 0..<K { acc += a[i*K+kk]*b[kk*N+j] }; if c[i*N+j] != acc { bad.append((i, j, c[i*N+j], acc)) } } }
            XCTAssertTrue(bad.isEmpty, "SGEMM \(M)x\(N)x\(K): \(bad.count) bad \(bad.prefix(4))")
        }
    }
    func testHistogramSmall() throws {
        let k = try PTXToMSLTranslator.translate(ptx: PTXTranslatorTests.ptx, entry: "histogram256")
        for (n, grid, block) in [(1000, 1, 256), (100000, 1, 256), (100000, 4, 256), (100000, 4, 128), (100000, 1, 32)] {
            let data = (0..<n).map { UInt8(truncatingIfNeeded: $0 &* 2654435761 >> 7) }
            let r = PTXRunner()
            let D = r.buffer(data), H = r.zeros(1024)
            try r.launch(k, grid: (grid, 1, 1), block: (block, 1, 1), [.ptr(D), .ptr(H), .i32(Int32(n))])
            let h = r.read(H, 256, as: UInt32.self)
            XCTAssertEqual(h.reduce(0, +), UInt32(n), "histogram n=\(n) grid=\(grid) block=\(block)")
        }
    }
}
