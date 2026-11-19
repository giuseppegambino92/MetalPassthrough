// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// PTX arrives from the (untrusted) guest. The translator must turn any input into either an error
// or MSL - never a Swift trap, which would take the whole VM's GPU service down. This test mutates
// real nvcc PTX thousands of times and runs every mutant through the translator (and a sample of the
// successful ones through the Metal compiler).

import MetalPassthroughHost
import Metal
import XCTest

final class PTXFuzzTests: XCTestCase {
    struct RNG {
        var s: UInt64
        mutating func next() -> UInt64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s }
        mutating func pick(_ n: Int) -> Int { n > 0 ? Int(next() % UInt64(n)) : 0 }
    }

    static let vocabulary = [
        "%r1", "%rd1", "%f1", "%fd1", "%p1", "%rs1", "%r999", "%nope", "%tid.x", "%ctaid.w", "%clock", "%laneid", "0", "-1",
        "99999999999999999999999", "0x", "0xFFFFFFFFFFFFFFFF", "0f", "0fZZZZZZZZ", "0d3FF0000000000000", "1.5e999", "[%rd1]",
        "[%rd1+-99999999999]", "[]", "[+4]", "{%r1,%r2}", "{}", "{%r1", "|", "!", "!%p1", "_", "$L__BB0_2", "$L__nowhere",
        "add.s32", "mad.wide.s32", "ld.global.v4.f32", "st.shared.u8", "atom.global.cas.b32", "shfl.sync.bfly.b32",
        "bar.sync", "cvt.rni.s64.f64", "setp.ltu.f64", "selp.b64", "prmt.b32", "lop3.b32", "bfe.s64", "bra", "ret", "call",
        "mma.sync", ".reg", ".reg .b32 %x<", ".reg .v4 .f32 %v;", ".shared .align 0 .b8 s[0]", ".local .b8 l[-1]", ";", ",", "(",
        ")", "@%p1", "@!%p9",
    ]

    func mutate(_ lines: [String], _ r: inout RNG) -> String {
        var l = lines
        for _ in 0..<(1 + r.pick(4)) {
            guard !l.isEmpty else { break }
            let i = r.pick(l.count)
            switch r.pick(8) {
            case 0: l.remove(at: i)
            case 1: l.insert(l[i], at: r.pick(l.count))
            case 2: l.swapAt(i, r.pick(l.count))
            case 3:
                var toks = l[i].split(separator: " ", omittingEmptySubsequences: false).map(String.init)
                if !toks.isEmpty { toks[r.pick(toks.count)] = Self.vocabulary[r.pick(Self.vocabulary.count)] }
                l[i] = toks.joined(separator: " ")
            case 4: l[i] += " " + Self.vocabulary[r.pick(Self.vocabulary.count)]
            case 5:
                let chars = Array(l[i])
                if !chars.isEmpty { l[i] = String(chars.prefix(r.pick(chars.count))) }
            case 6:
                let junk = (0..<r.pick(20)).map { _ in Character(UnicodeScalar(UInt8(32 + r.pick(95)))) }
                l.insert(String(junk), at: i)
            default:
                l = Array(l.prefix(r.pick(l.count)))  // truncated module
            }
        }
        return l.joined(separator: "\n")
    }

    func testMutatedPTXNeverTraps() throws {
        let lines = PTXTranslatorTests.ptx.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let entries = ["vector_add", "saxpy", "tiled_sgemm", "reduce_sum", "stencil2d", "histogram256", "mathmix", "daxpy"]
        let device = MTLCreateSystemDefaultDevice()!
        var r = RNG(s: 0xC0FFEE_1234_5678)
        var translated = 0, rejected = 0, compiled = 0, compileErrors = 0
        // 500 by default (fast CI); METALPASSTHROUGH_FUZZ_ITERS=4000 reproduces the long run.
        let iterations = Int(ProcessInfo.processInfo.environment["METALPASSTHROUGH_FUZZ_ITERS"] ?? "") ?? 500
        for it in 0..<iterations {
            let src = mutate(lines, &r)
            let entry = entries[r.pick(entries.count)]
            do {
                let k = try PTXToMSLTranslator.translate(ptx: src, entry: entry, options: .init(lowerFP64: r.pick(2) == 0))
                translated += 1
                if it % 50 == 0 {  // Metal compiles are slow: sample them
                    let o = MTLCompileOptions()
                    o.mathMode = .safe
                    if (try? device.makeLibrary(source: k.msl, options: o)) != nil { compiled += 1 } else { compileErrors += 1 }
                }
            } catch {
                rejected += 1
            }
        }
        print("PTX fuzz: \(iterations) mutants -> \(translated) translated, \(rejected) rejected with an error; "
              + "sampled Metal compiles: \(compiled) ok, \(compileErrors) compile errors")
        XCTAssertEqual(translated + rejected, iterations)
    }
}
