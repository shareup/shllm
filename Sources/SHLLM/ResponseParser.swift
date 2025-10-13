import Foundation
import class MLXLLM.GPTOSSModel
import class MLXLLM.Qwen2Model
import class MLXLLM.Qwen3Model
import class MLXLLM.Qwen3MoEModel
import enum MLXLMCommon.Generation
import Synchronized

public extension LLM {
    struct ResponseParser: Sendable {
        public var parse: @Sendable (Generation) -> Response?
    }
}

public extension LLM {
    static var defaultParser: ResponseParser {
        ResponseParser { (generation: Generation) -> Response? in
            switch generation {
            case let .chunk(chunk):
                return .text(chunk)

            case let .toolCall(toolCall):
                return .toolCall(toolCall)

            case .info:
                return nil
            }
        }
    }
}

public extension LLM where Model == Qwen2Model {
    static var deepSeekR1Parser: ResponseParser {
        // NOTE: DeepSeek R1 starts in thinking mode.
        let isThinking = Locked(true)
        let tokensToIgnore = Set(["<think>", "<think>\n"])
        let end = Set(["</think>", "</think>\n"])
        return ResponseParser { (generation: Generation) -> Response? in
            switch generation {
            case let .chunk(chunk):
                if tokensToIgnore.contains(chunk) {
                    return nil
                } else if end.contains(chunk) {
                    isThinking.access { $0 = false }
                    return nil
                } else if isThinking.access({ $0 }) {
                    return .reasoning(chunk)
                } else {
                    return .text(chunk)
                }

            case let .toolCall(toolCall):
                return .toolCall(toolCall)

            case .info:
                return nil
            }
        }
    }
}

public extension LLM where Model == Qwen3Model {
    static var qwen3Parser: ResponseParser = defaultThinkingParser
}

public extension LLM where Model == Qwen3MoEModel {
    static var qwen3MoEParser: ResponseParser = defaultThinkingParser
}

private extension LLM {
    static var defaultThinkingParser: ResponseParser {
        let isThinking = Locked(false)
        let start = Set(["<think>", "<think>\n"])
        let end = Set(["</think>", "</think>\n"])
        return ResponseParser { (generation: Generation) -> Response? in
            switch generation {
            case let .chunk(chunk):
                if start.contains(chunk) {
                    isThinking.access { $0 = true }
                    return nil
                } else if end.contains(chunk) {
                    isThinking.access { $0 = false }
                    return nil
                } else if isThinking.access({ $0 }) {
                    return .reasoning(chunk)
                } else {
                    return .text(chunk)
                }

            case let .toolCall(toolCall):
                return .toolCall(toolCall)

            case .info:
                return nil
            }
        }
    }
}
