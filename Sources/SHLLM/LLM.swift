import CoreGraphics
import Foundation
import struct Hub.Config
import Metal
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

public final class LLM {
    public static func clearCache() {
        MLX.GPU.clearCache()
    }

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

    static func cohere(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: CohereModel.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    static func gemma(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: GemmaModel.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    static func gemma2(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: Gemma2Model.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    static func internLM2(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: InternLM2Model.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    static func llama(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: LlamaModel.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    static func openELM(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: OpenELMModel.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    static func phi(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: PhiModel.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    static func phi3(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: Phi3Model.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    static func phiMoE(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: PhiMoEModel.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    static func qwen2(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: Qwen2Model.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    static func smolLM(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: LlamaModel.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    static func starcoder2(
        directory: URL,
        maxInputTokenLength: Int? = nil
    ) async throws -> LLM {
        try await Self(
            directory: directory,
            modelInit: Starcoder2Model.init,
            maxInputTokenLength: maxInputTokenLength
        )
    }

    private init<Configuration: Decodable>(
        directory: URL,
        modelInit: (Configuration) -> some LanguageModel,
        maxInputTokenLength: Int? = nil
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
                configuration: configuration,
                maxInputTokenLength: maxInputTokenLength
            ),
            tokenizer: tokenizer
        )
    }
}

extension LLM {
    func request<T: Codable>(
        tools: Tools,
        messages: [Message],
        maxOutputTokenCount: Int = 1024 * 1024
    ) async throws -> T {
        let result = try await request(
            .init(messages: messages, tools: tools.toSpec()),
            maxOutputTokenCount: maxOutputTokenCount
        )
        return try JSONDecoder().decode(
            T.self,
            from: Data(result.trimmingToolCallMarkup().utf8)
        )
    }

    func request(
        messages: [Message],
        maxOutputTokenCount: Int = 1024 * 1024
    ) async throws -> String {
        try await request(
            .init(messages: messages),
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    func request(
        _ input: UserInput,
        maxOutputTokenCount: Int = 1024 * 1024
    ) async throws -> String {
        let input = try await context.processor.prepare(input: input)

        let result = try MLXLMCommon.generate(
            input: input,
            parameters: .init(),
            context: context
        ) { tokens in
            if tokens.count >= maxOutputTokenCount {
                .stop
            } else {
                .more
            }
        }

        return result.output
    }
}

private extension String {
    func trimmingToolCallMarkup() -> String {
        let prefix = "<tool_call>"
        let suffix = "</tool_call>"

        let whitespace = CharacterSet.whitespacesAndNewlines
        var copy = trimmingCharacters(in: whitespace)
        copy.removeFirst(prefix.count)
        copy.removeLast(suffix.count)
        return copy.trimmingCharacters(in: whitespace)
    }
}
