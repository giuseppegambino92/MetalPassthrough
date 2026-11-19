// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

import CryptoKit
import Foundation
import Metal

public struct CompiledPipeline: @unchecked Sendable {
    public let state: MTLComputePipelineState
    public let cacheHit: Bool
}

public enum CompileError: Error, CustomStringConvertible {
    case library(String)
    case noFunction(String)
    case pipeline(String)

    public var description: String {
        switch self {
        case .library(let s): return "MSL compilation failed:\n\(s)"
        case .noFunction(let f): return "kernel function '\(f)' not found in library"
        case .pipeline(let s): return "pipeline creation failed: \(s)"
        }
    }
}

/// Process-wide compute pipeline cache: an in-memory map plus a persistent MTLBinaryArchive under
/// ~/Library/Caches/dev.metalpassthrough so backend (AIR -> GPU ISA) compiles survive container and
/// daemon restarts.
public final class PipelineCache: @unchecked Sendable {
    public let device: MTLDevice
    private let lock = NSLock()
    private var memory: [String: MTLComputePipelineState] = [:]
    private var archive: MTLBinaryArchive?
    private let archiveURL: URL
    public private(set) var compiles: UInt64 = 0
    public private(set) var hits: UInt64 = 0

    public init(device: MTLDevice) {
        self.device = device
        let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("dev.metalpassthrough", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        archiveURL = dir.appendingPathComponent("pipelines-\(device.registryID).binarchive")
        let d = MTLBinaryArchiveDescriptor()
        if FileManager.default.fileExists(atPath: archiveURL.path) { d.url = archiveURL }
        archive = (try? device.makeBinaryArchive(descriptor: d)) ?? {
            // Corrupt / incompatible archive (e.g. after an OS update): start fresh.
            try? FileManager.default.removeItem(at: archiveURL)
            return try? device.makeBinaryArchive(descriptor: MTLBinaryArchiveDescriptor())
        }()
    }

    public var stats: (compiles: UInt64, hits: UInt64, resident: Int) { lock.withLock { (compiles, hits, memory.count) } }

    public func pipeline(source: String, function: String, fastMath: Bool) throws -> CompiledPipeline {
        let key = SHA256.hash(data: Data((source + "\u{0}" + function + (fastMath ? "\u{0}F" : "\u{0}S")).utf8))
            .map { String(format: "%02x", $0) }.joined()
        if let ps = lock.withLock({ memory[key] }) {
            lock.withLock { hits += 1 }
            return CompiledPipeline(state: ps, cacheHit: true)
        }
        let opts = MTLCompileOptions()
        opts.mathMode = fastMath ? .fast : .safe
        opts.mathFloatingPointFunctions = fastMath ? .fast : .precise
        opts.languageVersion = .version3_2
        let lib: MTLLibrary
        do { lib = try device.makeLibrary(source: source, options: opts) } catch {
            throw CompileError.library((error as NSError).localizedDescription)
        }
        guard let fn = lib.makeFunction(name: function) else { throw CompileError.noFunction(function) }
        let desc = MTLComputePipelineDescriptor()
        desc.computeFunction = fn
        desc.label = function
        var archived = false
        if let archive {
            desc.binaryArchives = [archive]
            if let ps = try? device.makeComputePipelineState(
                descriptor: desc, options: .failOnBinaryArchiveMiss, reflection: nil)
            {
                lock.withLock { memory[key] = ps; hits += 1 }
                return CompiledPipeline(state: ps, cacheHit: true)
            }
            archived = (try? archive.addComputePipelineFunctions(descriptor: desc)) != nil
        }
        let ps: MTLComputePipelineState
        do { ps = try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil) } catch {
            throw CompileError.pipeline((error as NSError).localizedDescription)
        }
        lock.withLock { memory[key] = ps; compiles += 1 }
        if archived, let archive {
            // Several processes (one GPU worker per guest process) share this file: write a private
            // temporary and rename it into place atomically, so readers never see a torn archive.
            // Concurrent writers may drop each other's additions; that only costs a recompile.
            lock.withLock {
                let tmp = archiveURL.deletingLastPathComponent()
                    .appendingPathComponent(".\(archiveURL.lastPathComponent).\(getpid()).\(UInt64.random(in: 0...UInt64.max)).tmp")
                if (try? archive.serialize(to: tmp)) != nil {
                    if rename(tmp.path, archiveURL.path) != 0 { try? FileManager.default.removeItem(at: tmp) }
                }
            }
        }
        return CompiledPipeline(state: ps, cacheHit: false)
    }
}
