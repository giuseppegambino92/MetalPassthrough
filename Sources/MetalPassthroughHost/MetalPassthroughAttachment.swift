// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// One-stop integration point used by both `mpt-run` and the apple/container `--gpu` patch:
// reserves guest hugepages, attaches the MetalPassthrough virtio device + vsock endpoint to the VM
// and bind-mounts the guest runtime (libcuda, libcudart, libmetalpassthrough, apple-smi, Python).

import MPTProtocol
import Containerization
import Foundation

@available(macOS 27.0, *)
public final class MetalPassthroughAttachment: @unchecked Sendable {
    public let device: GuestMemoryDevice
    public let server: MetalPassthroughServer
    /// 2 MiB hugepages reserved at guest boot (the zero-copy arena pool).
    public let hugepages: Int
    public let guestRuntime: String?

    /// - Parameters:
    ///   - vmMemoryBytes: memory of the VM.
    ///   - gpuMemoryBytes: size of the GPU arena pool; default is half of the VM memory
    ///     (overridable with METALPASSTHROUGH_GPU_MEMORY_MB in the host environment).
    public init(containerID: String, vmMemoryBytes: UInt64, gpuMemoryBytes: UInt64? = nil, guestRuntime: String? = nil, verbose: Bool = false) throws {
        let envMB = ProcessInfo.processInfo.environment["METALPASSTHROUGH_GPU_MEMORY_MB"].flatMap { UInt64($0) }
        let pool = gpuMemoryBytes ?? envMB.map { $0 << 20 } ?? vmMemoryBytes / 2
        hugepages = max(16, Int(min(pool, vmMemoryBytes * 3 / 4) / UInt64(MPT_CHUNK_SIZE)))
        device = GuestMemoryDevice()
        server = try MetalPassthroughServer(memory: device, guestMemoryBytes: vmMemoryBytes, containerID: containerID, verbose: verbose)
        self.guestRuntime = guestRuntime.flatMap { $0.isEmpty ? nil : $0 } ?? Self.findGuestRuntime()
    }

    deinit { server.registry.remove() }

    /// Kernel command line additions (append to `Kernel.commandLine.kernelArgs`).
    public var kernelArgs: [String] { ["hugepages=\(hugepages)"] }

    /// The VZ extension to add to the VM configuration.
    public var vzExtension: MetalPassthroughVZExtension {
        let s = server
        return MetalPassthroughVZExtension(device: device) { conn in s.accept(conn) }
    }

    /// Wraps a VZ VM manager so every VM it creates carries the MetalPassthrough device.
    public func wrap(_ vmm: VZVirtualMachineManager) -> ExtendingVirtualMachineManager {
        ExtendingVirtualMachineManager(base: vmm, extensions: [vzExtension])
    }

    /// Bind-mounts the guest runtime at /usr/local/metalpassthrough and sets the loader / Python paths.
    public func configure(_ config: inout LinuxContainer.Configuration) {
        guard let rt = guestRuntime else { return }
        config.mounts.append(.share(source: rt, destination: "/usr/local/metalpassthrough", options: ["ro"]))
        var env = config.process.environmentVariables.filter { !$0.hasPrefix("PATH=") && !$0.hasPrefix("LD_LIBRARY_PATH=") }
        let path = config.process.environmentVariables.first { $0.hasPrefix("PATH=") }?.dropFirst(5)
            ?? "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        env += [
            "METALPASSTHROUGH=1",
            "PATH=/usr/local/metalpassthrough/bin:\(path)",
            "LD_LIBRARY_PATH=/usr/local/metalpassthrough/lib",
            "PYTHONPATH=/usr/local/metalpassthrough/python",
        ]
        config.process.environmentVariables = env
    }

    /// Guest runtime lookup: $METALPASSTHROUGH_GUEST_DIR, ./guest/dist (source checkout), <exe>/../share,
    /// ~/Library/Application Support/dev.metalpassthrough/guest (`make install`), then system prefixes.
    public static func findGuestRuntime() -> String? {
        var candidates: [String] = []
        if let e = ProcessInfo.processInfo.environment["METALPASSTHROUGH_GUEST_DIR"] { candidates.append(e) }
        // A source checkout wins when running from it (development); launchd helpers run with cwd "/".
        candidates.append(FileManager.default.currentDirectoryPath + "/guest/dist")
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        candidates += [exe.appendingPathComponent("../share/metalpassthrough/guest").path]
        candidates += [
            NSHomeDirectory() + "/Library/Application Support/dev.metalpassthrough/guest",
            "/usr/local/share/metalpassthrough/guest", "/opt/homebrew/share/metalpassthrough/guest",
        ]
        for c in candidates where FileManager.default.fileExists(atPath: c + "/lib/libmetalpassthrough.so.1") {
            return URL(fileURLWithPath: c).standardizedFileURL.path
        }
        return nil
    }
}
