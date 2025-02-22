import Foundation
import struct Hub.Config
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

public actor Phi3 {
    private let context: ModelContext
    private let configuration: ModelConfiguration

    public init() async throws {
        let decoder = JSONDecoder()

        let directory = try Self.directory
        let config = try Self.config

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

        let tokenizerConfig = try Self.tokenizerConfig
        let tokenizerData = try Self.tokenizerData

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

private extension Phi3 {
    static var directory: URL {
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

    static var config: Data {
        get throws {
            try Data(
                contentsOf: directory.appending(
                    path: "config.json",
                    directoryHint: .notDirectory
                )
            )
        }
    }

    static var tokenizerConfig: Config {
        get throws {
            let data = try Data(contentsOf: directory.appending(
                path: "tokenizer_config.json",
                directoryHint: .notDirectory
            ))

            guard let json = try JSONSerialization.jsonObject(
                with: data
            ) as? [NSString: Any] else {
                throw SHLLMError.invalidOrMissingConfig(
                    "tokenizer_config.json"
                )
            }

            return Config(json)
        }
    }

    static var tokenizerData: Config {
        get throws {
            let data = try Data(contentsOf: directory.appending(
                path: "tokenizer.json",
                directoryHint: .notDirectory
            ))

            guard let json = try JSONSerialization.jsonObject(
                with: data
            ) as? [NSString: Any] else {
                throw SHLLMError.invalidOrMissingConfig(
                    "tokenizer.json"
                )
            }

            return Config(json)
        }
    }
}
