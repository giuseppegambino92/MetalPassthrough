// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

import Containerization
import Foundation

struct FDWriter: Writer {
    let handle: FileHandle
    func write(_ data: Data) throws { try handle.write(contentsOf: data) }
    func close() throws {}
}
