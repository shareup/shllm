// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "SHLLM",
    platforms: [.iOS(.v16), .macOS(.v14)],
    products: [
        .library(
            name: "SHLLM",
            targets: ["SHLLM"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/shareup/mlx-swift-examples",
            from: "0.0.2"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            from: "0.1.17"
        ),
    ],
    targets: [
        .target(
            name: "SHLLM",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(
                    name: "Transformers",
                    package: "swift-transformers",
                    moduleAliases: ["Models": "TransformersModels"]
                ),
            ],
//            resources: [
//                .copy("Resources/DeepSeek-R1-Distill-Qwen-7B-4bit"),
//                .copy("Resources/gemma-2-2b-it-4bit"),
//                .copy("Resources/OpenELM-270M-Instruct"),
//                .copy("Resources/Phi-3.5-mini-instruct-4bit"),
//                .copy("Resources/Phi-3.5-MoE-instruct-4bit"),
//                .copy("Resources/Qwen1.5-0.5B-Chat-4bit"),
//                .copy("Resources/Qwen2.5-1.5B-Instruct-4bit"),
//                .copy("Resources/Qwen2.5-7B-Instruct-4bit"),
//            ],
            linkerSettings: [
                .linkedFramework("CoreGraphics", .when(platforms: [.macOS])),
            ]
        ),
        .testTarget(
            name: "SHLLMTests",
            dependencies: ["SHLLM"]
        ),
    ]
)
