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
            isThinking.access { isThinking -> Response? in
                switch generation {
                case let .chunk(chunk):
                    if tokensToIgnore.contains(chunk) {
                        return nil
                    } else if end.contains(chunk) {
                        isThinking = false
                        return nil
                    } else if isThinking {
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
}

public extension LLM where Model == Qwen3Model {
    static var qwen3Parser: ResponseParser = defaultThinkingParser
}

public extension LLM where Model == Qwen3MoEModel {
    static var qwen3MoEParser: ResponseParser = defaultThinkingParser
}

public extension LLM where Model == GPTOSSModel {
    static var gptOSSParser: ResponseParser {
        let parser = Locked(Harmony.StreamableParser(
            startingRole: .assistant
        ))
        let chunks = Locked([String]())
        return ResponseParser { (generation: Generation) -> Response? in
            parser.access { parser -> Response? in
                if case let .chunk(text) = generation {
                    chunks.access { $0.append(text) }
                } else if case .info = generation {
                    Swift.print("$$$", chunks.access { $0 })
                }

                // NOTE: LLMs are happy to go on and on even after responding
                //       with a tool call token. So, we need to check if the
                //       last token was a tool call token. If it was a tool
                //       call token, we need to stop inference.
                if let last = parser.tokens.last,
                   case let .special(s) = Harmony.Token(last),
                   s.isToolCall
                {
                    // NOTE: Stop inference if the last token produced
                    //       by the LLM was a tool call token.
                    return nil
                }

                do {
                    guard case let .chunk(token) = generation else {
                        switch generation {
                        case .chunk:
                            assertionFailure()
                            return nil
                        case let .toolCall(toolCall):
                            return .toolCall(toolCall)
                        case .info:
                            // NOTE: Stop inference after the LLM has
                            //       stopped producing tokens.
                            try? parser.processEOS()
                            return nil
                        }
                    }

                    let messageCount = parser.messages.count
                    try parser.process(token)

                    if let delta = parser.delta {
                        if parser.channel == "analysis" {
                            return .reasoning(delta)
                        } else if parser.channel == "final" {
                            return .text(delta)
                        }
                    }

                    guard parser.messages.count > messageCount,
                          let lastMessage = parser.messages.last,
                          lastMessage.author.role == .assistant,
                          let recipient = lastMessage.recipient,
                          recipient.hasPrefix("functions.")
                    else {
                        // NOTE: Continue inference because we are
                        //       expect more tokens.
                        return nil
                    }

                    let functionName = String(recipient.dropFirst("functions.".count))
                    let decoder = JSONDecoder()
                    if case let .text(content) = lastMessage.content.first,
                       let jsonData = content.data(using: .utf8),
                       let jsonObject = try? decoder.decode(JSONValue.self, from: jsonData),
                       let args = jsonObject.anyValue as? [String: Any]
                    {
                        let toolCall = ToolCall(
                            function: ToolCall.Function(
                                name: functionName,
                                arguments: args
                            )
                        )
                        try? parser.processEOS()
                        return .toolCall(toolCall)
                    }

                    // NOTE: Stop inference after seeing any tool call, even
                    //       if it's not valid.
                    try? parser.processEOS()
                    return nil
                } catch {
                    // NOTE: Stop inference after an error
                    try? parser.processEOS()
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
