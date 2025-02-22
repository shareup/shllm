import Foundation
import struct Hub.Config
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

public actor Phi3 {
    private let context: ModelContext
    private let configuration: ModelConfiguration
    private let directory: URL

    public init(directory dir: URL) async throws {
        directory = dir
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

        let phi3Config = try decoder.decode(
            Phi3Configuration.self,
            from: config
        )
        let model = Phi3Model(phi3Config)

        try loadWeights(
            modelDirectory: directory,
            model: model,
            quantization: baseConfig.quantization
        )

        guard let tokenizerConfigJSON = try JSONSerialization.jsonObject(
            with: try Data(contentsOf: directory.appending(
                path: "tokenizer_config.json",
                directoryHint: .notDirectory
            ))) as? [NSString: Any] else {
            throw SHLLMError.invalidOrMissingConfig(
                "tokenizer_config.json"
            )
        }

        let tokenizerConfig = Config(tokenizerConfigJSON)

        guard let tokenizerDataJSON = try JSONSerialization.jsonObject(
            with: try Data(contentsOf: directory.appending(
                path: "tokenizer.json",
                directoryHint: .notDirectory
            ))) as? [NSString: Any] else {
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

    public func request(
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

extension Phi3 {
    static var bundleDirectory: URL {
        get throws {
            guard let url = Bundle.shllm.url(
                forResource: "Phi-3.5-mini-instruct-4bit",
                withExtension: nil,
                subdirectory: "Resources"
            ) else {
                throw SHLLMError
                    .directoryNotFound("Phi-3.5-mini-instruct-4bit")
            }
            return url
        }
    }
}
