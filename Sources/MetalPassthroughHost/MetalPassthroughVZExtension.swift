// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

import Containerization
import ContainerizationExtras
import Foundation
@preconcurrency import Virtualization

/// Containerization `VZInstanceExtension` that attaches the MetalPassthrough virtio device to a
/// container's VM and opens the host-side vsock doorbell/handshake port.
@available(macOS 27.0, *)
public struct MetalPassthroughVZExtension: VZInstanceExtension {
    public static let defaultPort: UInt32 = 2049

    public let device: GuestMemoryDevice
    public let port: UInt32
    public let onConnection: @Sendable (FileHandle) -> Void

    public init(device: GuestMemoryDevice, port: UInt32 = Self.defaultPort, onConnection: @escaping @Sendable (FileHandle) -> Void) {
        self.device = device
        self.port = port
        self.onConnection = onConnection
    }

    public func configureVZ(
        _ config: inout VZVirtualMachineConfiguration,
        allocator: any AddressAllocator<Character>,
        storageDeviceCount: Int,
        mountsByID: [String: [Containerization.Mount]]
    ) throws {
        config.customVirtioDevices.append(device.makeConfiguration())
    }

    public func didCreate(_ instance: VZVirtualMachineInstance) throws {
        let listener = try instance.listen(port)
        let handler = onConnection
        Task.detached {
            for await conn in listener { handler(conn) }
        }
    }
}

/// Wraps `VZVirtualMachineManager` so that every VM it creates carries the given extensions.
/// (`LinuxContainer` does not forward `extensions` itself; `LinuxPod` does.)
public struct ExtendingVirtualMachineManager: VirtualMachineManager {
    public let base: VZVirtualMachineManager
    public let extensions: [any Sendable]

    public init(base: VZVirtualMachineManager, extensions: [any Sendable]) {
        self.base = base
        self.extensions = extensions
    }

    public func create(config: some VMCreationConfig) async throws -> any VirtualMachineInstance {
        var c = config.configuration
        c.extensions.append(contentsOf: extensions)
        return try base.create(config: StandardVMConfig(configuration: c))
    }
}
