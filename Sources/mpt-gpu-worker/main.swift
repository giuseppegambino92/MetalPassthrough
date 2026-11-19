// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// Per-session GPU worker: runs one guest process's MetalPassthrough session in its own host process,
// so its shaders get their own GPU address space. Spawned by the VM-owning process; not run by hand.
import Darwin
import MetalPassthroughHost

if #available(macOS 27.0, *) { exit(MetalPassthroughWorker.main()) }
exit(2)
