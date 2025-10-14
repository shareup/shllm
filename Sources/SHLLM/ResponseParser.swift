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

public extension LLM where Model == GPTOSSModel {
    fileprivate enum ParserToken: String, Sendable {
        case call = "<|call|>"
        case channel = "<|channel|>"
        case constrain = "<|constrain|>"
        case end = "<|end|>"
        case message = "<|message|>"
        case `return` = "<|return|>"
        case start = "<|start|>"
    }

    fileprivate enum ParserState: Sendable {
        case initial
        case channelStart
        case analysisChannel
        case commentaryChannel
        case finalChannel
        case finish

        mutating func parse(_ token: String) -> Response? {
            switch token {
            case .call:
                assertionFailure()
                return nil

            case .channel:
                self = .channelStart
                return nil

            case .constrain:
                assertionFailure()
                return nil

            case .end:
                self = .initial
                return nil

            case .message:
                assert(isInChannel)
                return nil

            case .return:
                self = .finish
                return nil

            case .start:
                return nil

            default:
                switch self {
                case .initial:
                    return nil

                case .channelStart:
                    switch token {
                    case "final":
                        self = .finalChannel
                        return nil
                    case "analysis":
                        self = .analysisChannel
                        return nil
                    case "commentary":
                        self = .commentaryChannel
                        return nil
                    default:
                        assertionFailure()
                        return .text(token)
                    }

                case .analysisChannel:
                    return .reasoning(token)

                case .commentaryChannel:
                    return .reasoning(token)

                case .finalChannel:
                    return .text(token)

                case .finish:
                    return nil
                }
            }
        }

        private var isInChannel: Bool {
            switch self {
            case .finalChannel, .analysisChannel, .commentaryChannel:
                true
            case .initial, .channelStart, .finish:
                false
            }
        }
    }

    static var gptOSSParser: ResponseParser {
        let parser = Locked(ParserState.initial)
        // TODO: Remove me!!!
        let text = Locked("")
        return ResponseParser { (generation: Generation) -> Response? in
            parser.access { parser -> Response? in
                if case let .chunk(string) = generation {
                    text.access { $0.append(string) }
                }

                switch generation {
                case let .chunk(chunk):
                    return parser.parse(chunk)

                case let .toolCall(toolCall):
                    return .toolCall(toolCall)

                case .info:
                    // TODO: Remove me!!!
                    Swift.print("$$$", text.access { $0 })
                    return nil
                }
            }
        }
    }
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

private func ~= (lhs: LLM<GPTOSSModel>.ParserToken, rhs: String) -> Bool {
    lhs.rawValue == rhs
}
