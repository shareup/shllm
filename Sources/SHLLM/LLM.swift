import AsyncAlgorithms
import CoreGraphics
import Foundation
import Metal
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import Tokenizers

public enum Response {
    case reasoning(String)
    case text(String)
    case toolCall(ToolCall)
}

public struct LLM<Model: LanguageModel>: AsyncSequence {
    public typealias Element = Response
    public typealias CustomConfiguration =
        (ModelConfiguration) -> ModelConfiguration

    private let directory: URL
    private let input: UserInput
    private let tools: [any ToolProtocol]
    private let maxInputTokenCount: Int?
    private let maxOutputTokenCount: Int?
    private let customConfiguration: CustomConfiguration?
    private let responseParser: ResponseParser

    public init(
        directory: URL,
        input: UserInput,
        processing: UserInput.Processing? = nil,
        tools: [any ToolProtocol] = [],
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil,
        customConfiguration: CustomConfiguration? = nil,
        responseParser: ResponseParser = Self.defaultParser
    ) {
        self.directory = directory
        let input = {
            var input = input
            if let processing {
                input.processing = processing
            }
            input.tools = tools.isEmpty
                ? nil
                : tools.map(\.schema)
            return input
        }()
        self.input = input
        self.tools = tools
        self.maxInputTokenCount = maxInputTokenCount
        self.maxOutputTokenCount = maxOutputTokenCount
        self.customConfiguration = customConfiguration
        self.responseParser = responseParser
    }

    public func makeAsyncIterator() -> AsyncIterator {
        Self.AsyncIterator(
            directory: directory,
            input: input,
            tools: tools,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount,
            customConfiguration: customConfiguration,
            responseParser: responseParser
        )
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        private let directory: URL
        private let input: UserInput
        private let tools: [any ToolProtocol]
        private let maxInputTokenCount: Int?
        private let maxOutputTokenCount: Int?
        private let customConfiguration: CustomConfiguration?
        private let responseParser: ResponseParser

        private var state: State

        private enum State {
            case initial
            case loaded(ModelContext)
            case streaming(
                AsyncStream<Generation>,
                AsyncStream<Generation>.AsyncIterator
            )
            case failed(Error)
            case finished
        }

        fileprivate init(
            directory: URL,
            input: UserInput,
            tools: [any ToolProtocol] = [],
            maxInputTokenCount: Int?,
            maxOutputTokenCount: Int?,
            customConfiguration: CustomConfiguration? = nil,
            responseParser: ResponseParser
        ) {
            self.directory = directory
            self.input = input
            self.maxInputTokenCount = maxInputTokenCount
            self.maxOutputTokenCount = maxOutputTokenCount
            self.customConfiguration = customConfiguration
            self.responseParser = responseParser
            self.tools = tools
            state = .initial
        }

        public mutating func next() async throws -> Response? {
            switch state {
            case .initial:
                do {
                    let context = try await loadModelContext(
                        directory: directory,
                        maxInputTokenCount: maxInputTokenCount,
                        customConfiguration: customConfiguration
                    )

                    state = .loaded(context)
                    return try await next()
                } catch {
                    state = .failed(error)
                    throw error
                }

            case let .loaded(context):
                let input = try await context.processor.prepare(input: input)
                let stream = try MLXLMCommon.generate(
                    input: input,
                    parameters: .init(maxTokens: maxOutputTokenCount),
                    context: context
                )

                var iterator = stream.makeAsyncIterator()

                repeat {
                    guard let next = await iterator.next() else {
                        state = .finished
                        return nil
                    }

                    guard let next = responseParser.parse(next) else {
                        continue
                    }

                    state = .streaming(stream, iterator)
                    return next
                } while true

            case let .failed(error):
                throw error

            case .finished:
                return nil

            case .streaming(let stream, var iterator):
                repeat {
                    guard let next = await iterator.next() else {
                        state = .finished
                        return nil
                    }

                    guard let next = responseParser.parse(next) else {
                        continue
                    }

                    state = .streaming(stream, iterator)
                    return next
                } while true
            }
        }
    }

    public var result: (reasoning: String?, text: String?, toolCalls: [ToolCall]?) {
        get async throws {
            var reasoning = ""
            var text = ""
            var toolCalls = [ToolCall]()

            for try await response in self {
                switch response {
                case let .reasoning(part):
                    reasoning += part
                case let .text(part):
                    text += part
                case let .toolCall(part):
                    toolCalls.append(part)
                }
            }

            let whitespace = CharacterSet.whitespacesAndNewlines
            reasoning = reasoning.trimmingCharacters(in: whitespace)
            text = text.trimmingCharacters(in: whitespace)

            return (
                reasoning: reasoning.isEmpty ? nil : reasoning,
                text: text.isEmpty ? nil : text,
                toolCalls: toolCalls.isEmpty ? nil : toolCalls
            )
        }
    }
}

public extension LLM {
    var text: TextAsyncSequence {
        TextAsyncSequence(llm: self)
    }

    struct TextAsyncSequence: AsyncSequence {
        public typealias Element = String
        public typealias CustomConfiguration = LLM<Model>.CustomConfiguration

        private let llm: LLM<Model>

        fileprivate init(llm: LLM<Model>) {
            self.llm = llm
        }

        public func makeAsyncIterator() -> AsyncIterator {
            AsyncIterator(llm.makeAsyncIterator())
        }

        public var result: String {
            get async throws {
                var text = ""
                for try await chunk in self {
                    text += chunk
                }
                return text
            }
        }

        public struct AsyncIterator: AsyncIteratorProtocol {
            private var iterator: LLM<Model>.AsyncIterator

            fileprivate init(_ iterator: LLM<Model>.AsyncIterator) {
                self.iterator = iterator
            }

            public mutating func next() async throws -> String? {
                while let response = try await iterator.next() {
                    switch response {
                    case .reasoning:
                        continue
                    case let .text(text):
                        return text
                    case .toolCall:
                        continue
                    }
                }
                return nil
            }
        }
    }
}

public extension LLM where Model: VLMModel {
    init(
        directory: URL,
        prompt: String,
        userMessage: String,
        image: URL,
        processing: UserInput.Processing? = nil,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil,
        customConfiguration: CustomConfiguration? = nil
    ) {
        let input = UserInput(chat: [
            .system(prompt),
            .user(userMessage, images: [.url(image)]),
        ])
        self.init(
            directory: directory,
            input: input,
            processing: processing,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount,
            customConfiguration: customConfiguration
        )
    }

    init(
        directory: URL,
        prompt: String,
        userMessage: String,
        image: Data,
        processing: UserInput.Processing? = nil,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil,
        customConfiguration: CustomConfiguration? = nil
    ) {
        let input = UserInput(chat: [
            .system(prompt),
            .user(
                userMessage,
                images: [.ciImage(.init(data: image)!)]
            ),
        ])
        self.init(
            directory: directory,
            input: input,
            processing: processing,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount,
            customConfiguration: customConfiguration
        )
    }
}

// MARK: - DeepSeek R1

extension LLM where Model == Qwen2Model {
    public static func deepSeekR1(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil,
        responseParser: ResponseParser = Self.deepSeekR1Parser
    ) throws -> LLM<Qwen2Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount,
            responseParser: responseParser
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

extension LLM where Model == GemmaModel {
    public static func gemma(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<GemmaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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

extension LLM where Model == Gemma2Model {
    public static func gemma2(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Gemma2Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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

    static var gemma2_9B: URL {
        get throws {
            let dir = "gemma-2-9b-it-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Gemma 3 Text

extension LLM where Model == Gemma3TextModel {
    public static func gemma3_1B(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Gemma3TextModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount,
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<end_of_turn>"]
                return config
            }
        )
    }

    static var gemma3_1B: URL {
        get throws {
            let dir = "gemma-3-1b-it-qat-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Gemma 3 Vision

extension LLM where Model == Gemma3 {
    public static func gemma3(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Gemma3> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount,
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<end_of_turn>"]
                return config
            }
        )
    }

    static var gemma3_4B_3Bit: URL {
        get throws {
            let dir = "gemma-3-4b-it-qat-3bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static var gemma3_4B_4Bit: URL {
        get throws {
            let dir = "gemma-3-4b-it-qat-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static var gemma3_12B: URL {
        get throws {
            let dir = "gemma-3-12b-it-qat-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static var gemma3_27B: URL {
        get throws {
            let dir = "gemma-3-27b-it-qat-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - GPT-OSS

extension LLM where Model == GPTOSSModel {
    public static func gptOSS(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil,
        responseParser: ResponseParser = Self.gptOSSParser
    ) throws -> LLM<GPTOSSModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount,
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<|call|>"]
                return config
            },
            responseParser: responseParser
        )
    }

    static var gptOSS_20B_8bit: URL {
        get throws {
            let dir = "gpt-oss-20b-MLX-8bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Llama

extension LLM where Model == LlamaModel {
    public static func codeLlama(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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

    public static func llama3(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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

    static var llama3_1__8B: URL {
        get throws {
            let dir = "Meta-Llama-3.1-8B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static var llama3_2__1B: URL {
        get throws {
            let dir = "Llama-3.2-1B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static var llama3_2__3B: URL {
        get throws {
            let dir = "Llama-3.2-3B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Mistral

extension LLM where Model == LlamaModel {
    public static func mistral(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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

    static var mistralNemo: URL {
        get throws {
            let dir = "Mistral-Nemo-Instruct-2407-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - OpenELM

extension LLM where Model == OpenELMModel {
    public static func openELM(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<OpenELMModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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

extension LLM where Model == PhiModel {
    public static func phi2(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<PhiModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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

extension LLM where Model == Phi3Model {
    public static func phi3(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Phi3Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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

extension LLM where Model == PhiMoEModel {
    public static func phiMoE(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<PhiMoEModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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

extension LLM where Model == Qwen2Model {
    public static func qwen1_5(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Qwen2Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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

    public static func qwen2_5(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<Qwen2Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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

    static var qwen2_5__7B: URL {
        get throws {
            let dir = "Qwen2.5-7B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Qwen3

extension LLM where Model == Qwen3Model {
    public static func qwen3(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil,
        responseParser: ResponseParser = Self.qwen3Parser
    ) throws -> LLM<Qwen3Model> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount,
            responseParser: responseParser
        )
    }

    static var qwen3__0_6B: URL {
        get throws {
            let dir = "Qwen3-0.6B-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static var qwen3__1_7B: URL {
        get throws {
            let dir = "Qwen3-1.7B-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static var qwen3_4B: URL {
        get throws {
            let dir = "Qwen3-4B-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }

    static var qwen3_8B: URL {
        get throws {
            let dir = "Qwen3-8B-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Qwen3 MoE

extension LLM where Model == Qwen3MoEModel {
    public static func qwen3MoE(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil,
        responseParser: ResponseParser = Self.qwen3MoEParser
    ) throws -> LLM<Qwen3MoEModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
            input: input,
            maxInputTokenCount: maxInputTokenCount,
            maxOutputTokenCount: maxOutputTokenCount,
            responseParser: responseParser
        )
    }

    static var qwen3MoE: URL {
        get throws {
            let dir = "Qwen3-30B-A3B-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}

// MARK: - Smol

extension LLM where Model == LlamaModel {
    public static func smolLM(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int? = nil,
        maxOutputTokenCount: Int? = nil
    ) throws -> LLM<LlamaModel> {
        try SHLLM.assertSupportedDevice
        return .init(
            directory: directory,
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
