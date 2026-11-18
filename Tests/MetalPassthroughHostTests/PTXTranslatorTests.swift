// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// Translates real nvcc-generated PTX (Fixtures/kernels.ptx, CUDA 13.5, sm_80) to MSL, compiles it
// with Metal and executes it on the local Apple GPU, comparing against CPU references.

import MetalPassthroughHost
import Foundation
import XCTest

final class PTXTranslatorTests: XCTestCase {
    static let ptx: String = {
        let url = Bundle.module.url(forResource: "kernels", withExtension: "ptx", subdirectory: "Fixtures")!
        return try! String(contentsOf: url, encoding: .utf8)
    }()

    func kernel(_ name: String, lowerFP64: Bool = false) throws -> TranslatedKernel {
        try PTXToMSLTranslator.translate(ptx: Self.ptx, entry: name, options: .init(lowerFP64: lowerFP64))
    }

    func rng(_ n: Int, seed: UInt64, scale: Float = 1) -> [Float] {
        var s = seed
        return (0..<n).map { _ in
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return (Float(UInt32(truncatingIfNeeded: s >> 33)) / Float(UInt32.max >> 1) - 1) * scale
        }
    }

    func testEntriesParse() throws {
        XCTAssertEqual(try PTXToMSLTranslator.entries(ptx: Self.ptx),
                       ["vector_add", "saxpy", "tiled_sgemm", "reduce_sum", "stencil2d", "histogram256", "mathmix", "daxpy"])
    }

    func testAllEntriesCompileOnMetal() throws {
        let r = PTXRunner()
        for k in try PTXToMSLTranslator.translateAll(ptx: Self.ptx) {
            XCTAssertNoThrow(try r.cache.pipeline(source: k.msl, function: k.mslName, fastMath: false), k.entry)
        }
    }

    func testVectorAddBitExact() throws {
        let k = try kernel("vector_add")
        XCTAssertEqual(k.params.map(\.offset), [0, 8, 16, 24])
        let n = (1 << 20) + 7
        let a = rng(n, seed: 1), b = rng(n, seed: 2)
        let r = PTXRunner()
        let A = r.buffer(a), B = r.buffer(b), C = r.zeros(n * 4)
        try r.launch(k, grid: ((n + 255) / 256, 1, 1), block: (256, 1, 1), [.ptr(A), .ptr(B), .ptr(C), .i32(Int32(n))])
        let c = r.read(C, n, as: Float.self)
        for i in 0..<n where c[i].bitPattern != (a[i] + b[i]).bitPattern { XCTFail("vector_add[\(i)]"); return }
    }

    func testSaxpyGridStrideBitExact() throws {
        let k = try kernel("saxpy")
        let n = 1_000_003
        let x = rng(n, seed: 3), y = rng(n, seed: 4)
        let alpha: Float = 1.7182817
        let r = PTXRunner()
        let X = r.buffer(x), Y = r.buffer(y)
        try r.launch(k, grid: (120, 1, 1), block: (256, 1, 1), [.i32(Int32(n)), .f32(alpha), .ptr(X), .ptr(Y)])
        let out = r.read(Y, n, as: Float.self)
        for i in 0..<n where out[i].bitPattern != y[i].addingProduct(x[i], alpha).bitPattern {
            XCTFail("saxpy[\(i)] \(out[i]) vs \(y[i].addingProduct(alpha, x[i]))"); return
        }
    }

    func testTiledSgemmBitExact() throws {
        let k = try kernel("tiled_sgemm")
        XCTAssertEqual(k.staticSharedBytes, 2048)
        let M = 257, N = 193, K = 131
        let a = rng(M * K, seed: 5), b = rng(K * N, seed: 6)
        let r = PTXRunner()
        let A = r.buffer(a), B = r.buffer(b), C = r.zeros(M * N * 4)
        try r.launch(k, grid: ((N + 15) / 16, (M + 15) / 16, 1), block: (16, 16, 1),
                     [.ptr(A), .ptr(B), .ptr(C), .i32(Int32(M)), .i32(Int32(N)), .i32(Int32(K))])
        let c = r.read(C, M * N, as: Float.self)
        for i in 0..<M {
            for j in 0..<N {
                var acc: Float = 0
                for kk in 0..<K { acc = acc.addingProduct(a[i * K + kk], b[kk * N + j]) }  // fma chain, same order as PTX
                if c[i * N + j].bitPattern != acc.bitPattern { XCTFail("sgemm[\(i),\(j)] \(c[i * N + j]) vs \(acc)"); return }
            }
        }
    }

    func testReduceSumWithShuffleAndAtomics() throws {
        let k = try kernel("reduce_sum")
        XCTAssertTrue(k.usesSharedMemory)
        let n = 3_000_001
        let x = rng(n, seed: 7)
        let r = PTXRunner()
        let X = r.buffer(x), O = r.zeros(16)
        try r.launch(k, grid: (64, 1, 1), block: (256, 1, 1), dynamicShared: 256 * 4, [.ptr(X), .ptr(O), .i32(Int32(n))])
        let got = Double(r.read(O, 1, as: Float.self)[0])
        let want = x.reduce(0.0) { $0 + Double($1) }
        XCTAssertEqual(got, want, accuracy: max(1e-3, abs(want) * 1e-5))
    }

    func testStencil2dBitExact() throws {
        let k = try kernel("stencil2d")
        let nx = 517, ny = 389
        let a = rng(nx * ny, seed: 8)
        let r = PTXRunner()
        let I = r.buffer(a), O = r.zeros(nx * ny * 4)
        try r.launch(k, grid: ((nx + 15) / 16, (ny + 15) / 16, 1), block: (16, 16, 1), [.ptr(I), .ptr(O), .i32(Int32(nx)), .i32(Int32(ny))])
        let o = r.read(O, nx * ny, as: Float.self)
        for y in 1..<(ny - 1) {
            for x in 1..<(nx - 1) {
                let i = y * nx + x
                let want: Float = 0.2 * ((((a[i] + a[i - 1]) + a[i + 1]) + a[i - nx]) + a[i + nx])
                if o[i].bitPattern != want.bitPattern { XCTFail("stencil[\(x),\(y)]"); return }
            }
        }
        XCTAssertEqual(o[0], 0)
    }

    func testHistogramSharedAndGlobalAtomics() throws {
        let k = try kernel("histogram256")
        let n = 4_000_037
        var s: UInt32 = 12345
        let data: [UInt8] = (0..<n).map { _ in s = s &* 1103515245 &+ 12345; return UInt8(truncatingIfNeeded: s >> 16) }
        let r = PTXRunner()
        let D = r.buffer(data), H = r.zeros(256 * 4)
        try r.launch(k, grid: (96, 1, 1), block: (256, 1, 1), [.ptr(D), .ptr(H), .i32(Int32(n))])
        var want = [UInt32](repeating: 0, count: 256)
        for d in data { want[Int(d)] += 1 }
        XCTAssertEqual(r.read(H, 256, as: UInt32.self), want)
    }

    func testMathMixIncludingLocalMemoryAndFP64Path() throws {
        let k = try kernel("mathmix")
        XCTAssertTrue(k.usesFP64)
        let n = 4096
        var x = rng(n, seed: 9, scale: 50)
        x[0] = 123456.75; x[1] = -987654.25; x[2] = 3.0e7  // force the Payne-Hanek (local memory + f64) path
        let r = PTXRunner()
        let X = r.buffer(x), Y = r.zeros(n * 4), Q = r.zeros(n * 4)
        try r.launch(k, grid: (n / 128, 1, 1), block: (128, 1, 1), [.ptr(X), .ptr(Y), .ptr(Q), .i32(Int32(n))])
        let y = r.read(Y, n, as: Float.self), q = r.read(Q, n, as: Int32.self)
        for i in 0..<n {
            let v = Double(x[i])
            let want = max(sin(v) * cos(v), -0.25) + sqrt(abs(v)) + exp(-v * v) + 1 / sqrt(1 + v * v) + log2(2 + v * v)
            XCTAssertEqual(Double(y[i]), want, accuracy: max(2e-4, abs(want) * 2e-5), "mathmix y[\(i)] x=\(x[i])")
            let ii = Int32(i)
            XCTAssertEqual(q[i], (ii * 7 - 3) / 5 + (ii % 3) - min(ii, 100) + max(ii >> 2, 3), "mathmix q[\(i)]")
        }
    }

    func testDaxpyDF64AndFP32Lowering() throws {
        let n = 100_000
        let x = (0..<n).map { Double($0) * 1.000000123456789 + 0.5 }
        let y = (0..<n).map { Double(n - $0) * 3.14159265358979 }
        let a = 2.718281828459045
        for (lower, tol) in [(false, 2e-13), (true, 2e-7)] {
            let k = try kernel("daxpy", lowerFP64: lower)
            XCTAssertEqual(k.fp64Lowered, lower)
            let r = PTXRunner()
            let X = r.buffer(x), Y = r.buffer(y)
            try r.launch(k, grid: ((n + 255) / 256, 1, 1), block: (256, 1, 1), [.i32(Int32(n)), .f64(a), .ptr(X), .ptr(Y)])
            let out = r.read(Y, n, as: Double.self)
            var worst = 0.0
            for i in 0..<n { worst = max(worst, abs(out[i] - (a * x[i] + y[i])) / abs(a * x[i] + y[i])) }
            XCTAssertLessThan(worst, tol, "daxpy lower=\(lower) worst relative error \(worst)")
            print("daxpy fp64 mode=\(lower ? "fp32" : "df64") worst relative error = \(worst)")
        }
    }
}
