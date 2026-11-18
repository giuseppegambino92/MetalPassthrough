// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

import ArgumentParser
import Containerization
import ContainerizationOCI
import MPTProtocol
import MetalPassthroughHost
import Foundation

@main
@available(macOS 27.0, *)
struct MPTRun: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mpt-run",
        abstract: "Run a Linux container (apple/containerization) with MetalPassthrough Metal passthrough attached."
    )

    @Option(name: .long, help: "Path to the guest Linux kernel")
    var kernel: String = NSHomeDirectory() + "/Library/Application Support/com.apple.container/kernels/default.kernel-arm64"

    @Option(name: .long, help: "vminitd init image reference")
    var initImage: String = "ghcr.io/apple/containerization/vminit:0.54.0"

    @Option(name: .long, help: "Number of vCPUs")
    var cpus: Int = 4

    @Option(name: .long, help: "VM memory in MiB")
    var memory: UInt64 = 8192

    @Option(name: .long, help: "2 MiB hugepages reserved at guest boot for the zero-copy GPU arena")
    var hugepages: Int = 1024

    @Option(name: [.customShort("v"), .long], help: "Bind mount host:guest (virtio-fs)")
    var volume: [String] = []

    @Option(name: [.customShort("e"), .long], help: "Environment variable KEY=VALUE")
    var env: [String] = []

    @Option(name: .long, help: "Working directory inside the container")
    var workdir: String = "/"

    @Option(name: .long, help: "Guest runtime directory bind-mounted at /usr/local/metalpassthrough (lib/, bin/, python/)")
    var runtime: String = ""

    @Option(name: .long, help: "Containerization image store (default: apple/container's store when present)")
    var imageStore: String = {
        let p = NSHomeDirectory() + "/Library/Application Support/com.apple.container"
        return FileManager.default.fileExists(atPath: p + "/content") ? p : ""
    }()

    @Flag(name: .long, help: "Do not attach a network interface")
    var noNetwork = false

    @Flag(name: .long, help: "Verbose host-side logging")
    var verbose = false

    @Argument(help: "Image reference")
    var image: String

    @Argument(parsing: .captureForPassthrough, help: "Command to run")
    var command: [String] = []

    /// Unpacks each image once into ~/Library/Caches/dev.metalpassthrough/rootfs/<digest>.ext4 and
    /// gives every container an APFS copy-on-write clone (instant, no extra disk until written).
    static func cachedRootfs(_ img: Containerization.Image, sizeBytes: UInt64, containerID: String) async throws -> URL {
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("dev.metalpassthrough/rootfs", isDirectory: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let digest = img.digest.replacingOccurrences(of: ":", with: "-")
        let base = cache.appendingPathComponent("\(digest).ext4")
        if !FileManager.default.fileExists(atPath: base.path) {
            let tmp = cache.appendingPathComponent("\(digest).\(getpid()).partial")
            try? FileManager.default.removeItem(at: tmp)
            _ = try await EXT4Unpacker(capacityInBytes: sizeBytes).unpack(img, for: .current, at: tmp)
            try FileManager.default.moveItem(at: tmp, to: base)
        }
        let dir = SessionRegistry.root.appendingPathComponent(containerID, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let clone = dir.appendingPathComponent("rootfs.ext4")
        guard clonefile(base.path, clone.path, 0) == 0 else {
            throw ValidationError("clonefile(\(base.path)) failed: \(String(cString: strerror(errno)))")
        }
        return clone
    }

    func run() async throws {
        let id = "mpt-" + UUID().uuidString.prefix(8).lowercased()
        var k = Kernel(path: URL(fileURLWithPath: kernel), platform: .linuxArm)

        let storeRoot = URL(fileURLWithPath: NSHomeDirectory() + "/Library/Application Support/dev.metalpassthrough")
        try FileManager.default.createDirectory(at: storeRoot, withIntermediateDirectories: true)
        // Share apple/container's image store so images built/pulled with `container` are usable.
        let store = try ImageStore(path: URL(fileURLWithPath: imageStore.isEmpty ? storeRoot.path : imageStore))
        let initPath = storeRoot.appendingPathComponent("initfs-0.54.0.ext4")
        let initfs: Containerization.Mount
        if FileManager.default.fileExists(atPath: initPath.path) {
            initfs = .block(format: "ext4", source: initPath.path, destination: "/", options: ["ro"])
        } else {
            let ii = try await store.getInitImage(reference: initImage)
            initfs = try await ii.initBlock(at: initPath, for: .linuxArm)
        }

        let gpu = try MetalPassthroughAttachment(
            containerID: id, vmMemoryBytes: memory << 20, gpuMemoryBytes: UInt64(hugepages) * UInt64(MPT_CHUNK_SIZE),
            guestRuntime: runtime, verbose: verbose)
        k.commandLine.kernelArgs += gpu.kernelArgs
        let info = gpu.server.info
        if verbose {
            FileHandle.standardError.write(
                "[mpt-run] \(info.name) (\(info.gpuCores)-core GPU, \(info.metalFamily)), \(String(format: "%.0f", info.measuredBandwidthGBs)) GB/s measured; VM \(cpus) vCPU / \(memory) MiB, \(gpu.hugepages) x 2 MiB hugepages, guest runtime \(gpu.guestRuntime ?? "<none>")\n"
                    .data(using: .utf8)!)
        }
        let vmm = gpu.wrap(VZVirtualMachineManager(kernel: k, initialFilesystem: initfs))

        // Networking is optional for GPU workloads; continue without it if vmnet is unavailable.
        var network: (any Network)?
        if !noNetwork {
            do { network = try VmnetNetwork() } catch {
                FileHandle.standardError.write("[mpt-run] warning: networking disabled (\(error))\n".data(using: .utf8)!)
            }
        }
        var manager = try ContainerManager(vmm: vmm, network: network)
        let ref = image.contains("/") ? image : "docker.io/library/" + image
        let img = try await store.get(reference: ref, pull: !ref.hasPrefix("docker.io/library/metalpassthrough-"))
        let args = command
        let rootfs = try await Self.cachedRootfs(img, sizeBytes: 8 << 30, containerID: id)
        // ContainerManager writes the boot log under <store>/containers/<id>/ (only created by its unpack path).
        try FileManager.default.createDirectory(
            at: ImageStore.default.path.appendingPathComponent("containers").appendingPathComponent(id), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: rootfs.deletingLastPathComponent()) }
        let container = try await manager.create(
            id, image: img, rootfs: .block(format: "ext4", source: rootfs.path, destination: "/"), networking: network != nil,
            vm: VMResources(cpus: cpus, memoryInBytes: (memory << 20) + VMResources.guestMemoryOverhead)
        ) { config in
            config.cpus = cpus
            config.memoryInBytes = memory << 20
            if !args.isEmpty { config.process.arguments = args }
            config.process.workingDirectory = workdir
            config.process.environmentVariables += env
            gpu.configure(&config)
            config.process.stdout = FDWriter(handle: .standardOutput)
            config.process.stderr = FDWriter(handle: .standardError)
            for v in volume {
                let p = v.split(separator: ":", maxSplits: 1).map(String.init)
                guard p.count == 2 else { throw ValidationError("bad --volume \(v)") }
                config.mounts.append(.share(source: p[0], destination: p[1]))
            }
        }
        defer { try? manager.delete(id) }
        try await container.create()
        try await container.start()
        let status = try await container.wait()
        try await container.stop()
        if status.exitCode != 0 { throw ExitCode(status.exitCode) }
    }
}
