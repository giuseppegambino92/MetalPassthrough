// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

import Darwin
import Foundation

/// Minimal blocking, exact-length I/O on a raw file descriptor (vsock / unix stream sockets).
public struct FDStream: Sendable {
    public let fd: Int32
    public init(fd: Int32) { self.fd = fd }

    public enum IOError: Error { case eof, errno(Int32) }

    public func readExact(_ buf: UnsafeMutableRawPointer, _ n: Int) throws {
        var off = 0
        while off < n {
            let r = Darwin.read(fd, buf + off, n - off)
            if r == 0 { throw IOError.eof }
            if r < 0 { if errno == EINTR { continue }; throw IOError.errno(errno) }
            off += r
        }
    }

    public func readExact(_ n: Int) throws -> [UInt8] {
        var out = [UInt8](repeating: 0, count: n)
        try out.withUnsafeMutableBytes { try readExact($0.baseAddress!, n) }
        return out
    }

    public func writeAll(_ buf: UnsafeRawPointer, _ n: Int) throws {
        var off = 0
        while off < n {
            let r = Darwin.write(fd, buf + off, n - off)
            if r < 0 { if errno == EINTR { continue }; throw IOError.errno(errno) }
            off += r
        }
    }

    public func writeAll(_ bytes: [UInt8]) throws {
        try bytes.withUnsafeBytes { try writeAll($0.baseAddress!, bytes.count) }
    }
}
