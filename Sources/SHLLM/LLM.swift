import AsyncAlgorithms
import CoreGraphics
import Foundation
import struct Hub.Config
import Metal
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

public struct LLM<
    Configuration: Decodable,
    Model: LanguageModel
>: AsyncSequence {
    public typealias Element = String

    private let directory: URL
    private let modelInit: (Configuration) -> Model
    private let input: UserInput
    private let maxInputTokenCount: Int?
    private let maxOutputTokenCount: Int?

    init(
        directory: URL,
        modelInit: @escaping (Configuration) -> Model,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) {
        self.directory = directory
        self.modelInit = modelInit
        self.input = input
        self.maxInputTokenCount = maxInputTokenCount
        self.maxOutputTokenCount = maxOutputTokenCount
    }

    public func makeAsyncIterator() -> AsyncIterator {
        Self.AsyncIterator(
            directory: directory,
            modelInit: modelInit,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        private let directory: URL
        private let input: UserInput
        private let maxInputTokenCount: Int?
        private let maxOutputTokenCount: Int?

        private var state: State

        private enum State {
            case initialized(ModelContext)
            case streaming(
                AsyncStream<Generation>,
                AsyncStream<Generation>.AsyncIterator,
                NaiveStreamingDetokenizer,
                Int
            )
            case failed(Error)
            case finished
        }

        fileprivate init(
            directory: URL,
            modelInit: (Configuration) -> some LanguageModel,
            input: UserInput,
            maxInputTokenCount: Int?,
            maxOutputTokenCount: Int?
        ) {
            self.directory = directory
            self.input = input
            self.maxInputTokenCount = maxInputTokenCount
            self.maxOutputTokenCount = maxOutputTokenCount

            let decoder = JSONDecoder()

            do {
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

                let configuration = ModelConfiguration(
                    directory: directory,
                    overrideTokenizer: nil,
                    defaultPrompt: "You are a helpful assistant."
                )

                let context = ModelContext(
                    configuration: configuration,
                    model: model,
                    processor: LLMUserInputProcessor(
                        tokenizer: tokenizer,
                        configuration: configuration,
                        maxInputTokenCount: maxInputTokenCount
                    ),
                    tokenizer: tokenizer
                )

                state = .initialized(context)
            } catch {
                state = .failed(error)
            }
        }

        public mutating func next() async throws -> String? {
            switch state {
            case let .initialized(context):
                let input = try await context.processor.prepare(input: input)
                let stream = try MLXLMCommon.generate(
                    input: input,
                    parameters: .init(),
                    context: context
                )

                var iterator = stream.makeAsyncIterator()
                var detokenizer = NaiveStreamingDetokenizer(
                    tokenizer: context.tokenizer
                )
                var tokenCount = 0

                repeat {
                    if let maxOutputTokenCount, tokenCount >= maxOutputTokenCount {
                        state = .finished
                        return nil
                    }

                    guard case let .token(token) = await iterator.next() else {
                        state = .finished
                        return nil
                    }

                    tokenCount += 1
                    detokenizer.append(token: token)

                    if let output = detokenizer.next() {
                        state = .streaming(
                            stream,
                            iterator,
                            detokenizer,
                            tokenCount
                        )
                        return output
                    }
                } while true

            case let .failed(error):
                throw error

            case .finished:
                return nil

            case .streaming(
                let stream,
                var iterator,
                var detokenizer,
                var tokenCount
            ):
                repeat {
                    if let maxOutputTokenCount, tokenCount >= maxOutputTokenCount {
                        state = .finished
                        return nil
                    }

                    guard case let .token(token) = await iterator.next() else {
                        state = .finished
                        return nil
                    }

                    tokenCount += 1
                    detokenizer.append(token: token)

                    if let output = detokenizer.next() {
                        state = .streaming(
                            stream,
                            iterator,
                            detokenizer,
                            tokenCount
                        )
                        return output
                    }
                } while true
            }
        }
    }
}

// MARK: - DeepSeek R1

public extension LLM where Configuration == Qwen2Configuration, Model == Qwen2Model {
    static func deepSeekR1(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Qwen2Configuration, Qwen2Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: Qwen2Model.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var deepSeekR1: URL {
        get throws {
            let dir = "DeepSeek-R1-Distill-Qwen-7B-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Gemma

public extension LLM where Configuration == GemmaConfiguration, Model == GemmaModel {
    static func gemma(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<GemmaConfiguration, GemmaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: GemmaModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var gemma: URL {
        get throws {
            let dir = "quantized-gemma-2b-it"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Gemma 2

public extension LLM where Configuration == Gemma2Configuration, Model == Gemma2Model {
    static func gemma2_2B(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Gemma2Configuration, Gemma2Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: Gemma2Model.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var gemma2_2B: URL {
        get throws {
            let dir = "gemma-2-2b-it-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static func gemma2_9B(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Gemma2Configuration, Gemma2Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: Gemma2Model.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var gemma2_9B: URL {
        get throws {
            let dir = "gemma-2-9b-it-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Llama

public extension LLM where Configuration == LlamaConfiguration, Model == LlamaModel {
    static func codeLlama(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaConfiguration, LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: LlamaModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var codeLlama: URL {
        get throws {
            let dir = "CodeLlama-13b-Instruct-hf-4bit-MLX"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static func llama3_8B(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaConfiguration, LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: LlamaModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var llama3_8B: URL {
        get throws {
            let dir = "Meta-Llama-3-8B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static func llama3_1__8B(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaConfiguration, LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: LlamaModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var llama3_1__8B: URL {
        get throws {
            let dir = "Meta-Llama-3.1-8B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static func llama3_2__1B(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaConfiguration, LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: LlamaModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var llama3_2__1B: URL {
        get throws {
            let dir = "Llama-3.2-1B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static func llama3_2__3B(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaConfiguration, LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: LlamaModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var llama3_2__3B: URL {
        get throws {
            let dir = "Llama-3.2-3B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Mistral

public extension LLM where Configuration == LlamaConfiguration, Model == LlamaModel {
    static func mistral7B(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaConfiguration, LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: LlamaModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var mistral7B: URL {
        get throws {
            let dir = "Mistral-7B-Instruct-v0.3-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static func mistralNemo(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaConfiguration, LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: LlamaModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var mistralNemo: URL {
        get throws {
            let dir = "Mistral-Nemo-Instruct-2407-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - OpenELM

public extension LLM where Configuration == OpenElmConfiguration, Model == OpenELMModel {
    static func openELM(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<OpenElmConfiguration, OpenELMModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: OpenELMModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var openELM: URL {
        get throws {
            let dir = "OpenELM-270M-Instruct"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Phi

public extension LLM where Configuration == PhiConfiguration, Model == PhiModel {
    static func phi2(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<PhiConfiguration, PhiModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: PhiModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var phi2: URL {
        get throws {
            let dir = "phi-2-hf-4bit-mlx"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Phi 3

public extension LLM where Configuration == Phi3Configuration, Model == Phi3Model {
    static func phi3(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Phi3Configuration, Phi3Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: Phi3Model.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var phi3: URL {
        get throws {
            let dir = "Phi-3.5-mini-instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Phi MoE

public extension LLM where Configuration == PhiMoEConfiguration, Model == PhiMoEModel {
    static func phiMoE(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<PhiMoEConfiguration, PhiMoEModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: PhiMoEModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var phiMoE: URL {
        get throws {
            let dir = "Phi-3.5-MoE-instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Qwen

public extension LLM where Configuration == Qwen2Configuration, Model == Qwen2Model {
    static func qwen1_5(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Qwen2Configuration, Qwen2Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: Qwen2Model.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var qwen1_5: URL {
        get throws {
            let dir = "Qwen1.5-0.5B-Chat-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static func qwen2_5__1_5B(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Qwen2Configuration, Qwen2Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: Qwen2Model.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var qwen2_5__1_5B: URL {
        get throws {
            let dir = "Qwen2.5-1.5B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static func qwen2_5__7B(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Qwen2Configuration, Qwen2Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: Qwen2Model.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var qwen2_5__7B: URL {
        get throws {
            let dir = "Qwen2.5-7B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Smol

public extension LLM where Configuration == LlamaConfiguration, Model == LlamaModel {
    static func smolLM(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaConfiguration, LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            modelInit: LlamaModel.init,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount
        )
    }

    static var smolLM: URL {
        get throws {
            let dir = "SmolLM-135M-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}
