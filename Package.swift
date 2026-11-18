// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "MetalPassthrough",
    platforms: [.macOS("15.0")],
    products: [
        .library(name: "MetalPassthroughHost", targets: ["MetalPassthroughHost"]),
        .executable(name: "mpt-run", targets: ["mpt-run"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/containerization.git", exact: "0.54.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.7.0"),
    ],
    targets: [
        .target(name: "MPTProtocol"),
        .target(
            name: "MetalPassthroughHost",
            dependencies: [
                "MPTProtocol",
                .product(name: "Containerization", package: "containerization"),
            ],
            linkerSettings: [.linkedFramework("Metal"), .linkedFramework("Virtualization")]
        ),
        .executableTarget(
            name: "mpt-run",
            dependencies: [
                "MetalPassthroughHost",
                "MPTProtocol",
                .product(name: "Containerization", package: "containerization"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "MetalPassthroughHostTests",
            dependencies: ["MetalPassthroughHost", "MPTProtocol"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
