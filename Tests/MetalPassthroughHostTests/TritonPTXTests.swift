// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// PTX produced by Triton 3.5.1 (ahead-of-time, target cuda sm_80, num_warps 4; see
// e2e/triton/compile_ptx.py) translated to Metal and executed on the local Apple GPU.

import MetalPassthroughHost
import XCTest

final class TritonPTXTests: XCTestCase {
    func fixture(_ name: String) throws -> String {
        let url = Bundle.module.url(forResource: name, withExtension: "ptx", subdirectory: "Fixtures")!
        return try String(contentsOf: url, encoding: .utf8)
    }

    func testTritonVectorAdd() throws {
        let k = try PTXToMSLTranslator.translate(ptx: fixture("add_kernel"), entry: "add_kernel")
        XCTAssertEqual(k.params.count, 6)  // x, y, out, n + Triton's global/profile scratch pointers
        if let d = ProcessInfo.processInfo.environment["MPT_DUMP_MSL"] { try? k.msl.write(toFile: d, atomically: true, encoding: .utf8) }
        let n = 1_000_003
        let x = (0..<n).map { Float($0) * 0.5 }, y = (0..<n).map { Float(n - $0) }
        let r = PTXRunner()
        let X = r.buffer(x), Y = r.buffer(y), O = r.zeros(n * 4)
        try r.launch(k, grid: ((n + 1023) / 1024, 1, 1), block: (128, 1, 1),
                     [.ptr(X), .ptr(Y), .ptr(O), .i32(Int32(n)), .u64(0), .u64(0)])
        let o = r.read(O, n, as: Float.self)
        if ProcessInfo.processInfo.environment["MPT_DUMP_MSL"] != nil {
            for i in [0, 1, 2, 3, 127, 128, 129, 1023, 1024, 1025] { print("o[\(i)] = \(o[i])  x=\(x[i]) y=\(y[i]) want \(x[i] + y[i])") }
        }
        for i in 0..<n where o[i] != x[i] + y[i] { XCTFail("add_kernel[\(i)] \(o[i]) != \(x[i] + y[i])"); return }
    }

    func testTritonSoftmax() throws {
        let k = try PTXToMSLTranslator.translate(ptx: fixture("softmax_kernel"), entry: "softmax_kernel")
        let rows = 300, cols = 781
        var s: UInt64 = 42
        let a: [Float] = (0..<rows * cols).map { _ in s = s &* 6364136223846793005 &+ 1; return Float(Int64(s >> 40) % 2000) / 100 - 10 }
        let r = PTXRunner()
        let I = r.buffer(a), O = r.zeros(rows * cols * 4)
        try r.launch(k, grid: (rows, 1, 1), block: (128, 1, 1), dynamicShared: Int(k.staticSharedBytes),
                     [.ptr(O), .ptr(I), .i32(Int32(cols)), .i32(Int32(cols)), .u64(0), .u64(0)])
        let o = r.read(O, rows * cols, as: Float.self)
        var worst = 0.0
        for row in 0..<rows {
            let v = a[row * cols..<(row + 1) * cols].map(Double.init)
            let m = v.max()!
            let e = v.map { Foundation.exp($0 - m) }
            let sum = e.reduce(0, +)
            for c in 0..<cols { worst = max(worst, abs(Double(o[row * cols + c]) - e[c] / sum)) }
        }
        print("Triton softmax \(rows)x\(cols): max abs error vs float64 \(worst)")
        XCTAssertLessThan(worst, 1e-6)
    }
}
