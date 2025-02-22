// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "SHLLM",
    platforms: [.iOS(.v16), .macOS(.v14)],
    products: [
        // Products define the executables and libraries a package produces, making them visible
        // to other packages.
        .library(
            name: "SHLLM",
            targets: ["SHLLM"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-examples/", branch: "main"),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMinor(from: "0.1.17")
        ),
    ],
    targets: [
        .target(
            name: "SHLLM",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            resources: [
                .copy("Resources"),
            ]
        ),
        .testTarget(
            name: "SHLLMTests",
            dependencies: ["SHLLM"]
        ),
    ]
)
