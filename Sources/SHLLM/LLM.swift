import CoreGraphics
import Foundation
import struct Hub.Config
import Metal
import MLX
import MLXLLM
import MLXNN
import MLXLMCommon
import Tokenizers

public final class LLM {
    public static var isSupportedDevice: Bool {
        guard let _ = MTLCreateSystemDefaultDevice() else {
            return false
        }
        return true
    }

    static var assertSupportedDevice: Void {
        get throws {
            guard isSupportedDevice else {
                throw SHLLMError.unsupportedDevice
            }
        }
    }

    private let directory: URL
    private let context: ModelContext
    private let configuration: ModelConfiguration

    static func cohere(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: CohereModel.init
        )
    }

    static func gemma(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: GemmaModel.init
        )
    }

    static func gemma2(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: Gemma2Model.init
        )
    }

    static func internLM2(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: InternLM2Model.init
        )
    }

    static func llama(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: LlamaModel.init
        )
    }

    static func openELM(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: OpenELMModel.init
        )
    }

    static func phi(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: PhiModel.init
        )
    }

    static func phi3(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: Phi3Model.init
        )
    }

    static func phiMoE(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: PhiMoEModel.init
        )
    }

    static func qwen2(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: Qwen2Model.init
        )
    }

    static func smolLM(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: LlamaModel.init
        )
    }

    static func starcoder2(directory: URL) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: Starcoder2Model.init
        )
    }

    private init<Configuration: Decodable>(
        directory: URL,
        modelInit: (Configuration) -> some LanguageModel
    ) async throws {
        self.directory = directory
        let decoder = JSONDecoder()

        let config = try Data(
            contentsOf: directory.appending(
                path: "config.json",
                directoryHint: .notDirectory
            )
        )

        let baseConfig = try decoder.decode(
            BaseConfiguration.self,
            from: config
        )

        let modelConfig = try decoder.decode(
            Configuration.self,
            from: config
        )
        let model = modelInit(modelConfig)

        try loadWeights(
            modelDirectory: directory,
            model: model,
            quantization: baseConfig.quantization
        )

        guard let tokenizerConfigJSON = try JSONSerialization.jsonObject(
            with: try Data(contentsOf: directory.appending(
                path: "tokenizer_config.json",
                directoryHint: .notDirectory
            ))
        ) as? [NSString: Any] else {
            throw SHLLMError.invalidOrMissingConfig(
                "tokenizer_config.json"
            )
        }

        let tokenizerConfig = Config(tokenizerConfigJSON)

        guard let tokenizerDataJSON = try JSONSerialization.jsonObject(
            with: try Data(contentsOf: directory.appending(
                path: "tokenizer.json",
                directoryHint: .notDirectory
            ))
        ) as? [NSString: Any] else {
            throw SHLLMError.invalidOrMissingConfig(
                "tokenizer.json"
            )
        }

        let tokenizerData = Config(tokenizerDataJSON)

        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: tokenizerConfig,
            tokenizerData: tokenizerData
        )

        configuration = ModelConfiguration(
            directory: directory,
            overrideTokenizer: nil,
            defaultPrompt: "You are a helpful assistant."
        )

        context = ModelContext(
            configuration: configuration,
            model: model,
            processor: LLMUserInputProcessor(
                tokenizer: tokenizer,
                configuration: configuration
            ),
            tokenizer: tokenizer
        )
    }
}

extension LLM {
    func request(
        _ input: UserInput,
        maxTokenCount: Int = 1024 * 1024
    ) async throws -> String {
        let input = try await context.processor.prepare(input: input)

        let result = try MLXLMCommon.generate(
            input: input,
            parameters: .init(),
            context: context
        ) { tokens in
            if tokens.count >= maxTokenCount {
                .stop
            } else {
                .more
            }
        }

        return result.output
    }
}

// FROM: https://github.com/ml-explore/mlx-swift-examples/blob/20701c0eeedd339ede4dd3b964152d814a3e9716/Libraries/MLXLMCommon/Load.swift#L61
private func loadWeights(
    modelDirectory: URL, model: LanguageModel, quantization: BaseConfiguration.Quantization? = nil
) throws {
    // load the weights
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }

    // per-model cleanup
    weights = model.sanitize(weights: weights)

    // quantize if needed
    if let quantization {
        quantize(model: model, groupSize: quantization.groupSize, bits: quantization.bits) {
            path, module in
            weights["\(path).scales"] != nil
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    // NOTE: removed verify: [.all] becuase Qwen models are not ready for that verification yet
    try model.update(parameters: parameters, verify: [])

    eval(model)
}
