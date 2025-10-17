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
        let state = Locked(State())
        return ResponseParser { (generation: Generation) -> Response? in
            state.access { state -> Response? in
                // NOTE: Because MLX Swift does not natively support
                //       Harmony, the tool calls produced by GPT-OSS
                //       are not sent as `Generation.toolCall` and
                //       the `<|call|>` token is not recognized as
                //       a stop token. In order to make tool call
                //       work in SHLLM, we manually parse the
                //       Harmony message format and extract the tool
                //       call from the message. However, since it's
                //       incorrect to continue inference after the
                //       model produces `<|call|>`, we added that
                //       token to `extraEOSTokens`, which means that
                //       MLX Swift will stop generating tokens when it
                //       encounters `<|call|>`. So, our Harmony parser
                //       will never actually see the tool call token, which
                //       means we won't know when to send a tool call.
                //       To work around this, we check for the presence of
                //       a tool call after MLX Swift stops generating tokens.
                //       If one exists, we send it to the client. But, to
                //       prevent a loop where we send the same tool call
                //       over and over again, we need to break if we've
                //       already sent the tool call.
                //
                //       The fix for this will be to add Harmony support
                //       directly to MLX Swift. At the very least, we'll
                //       need to add a new `ToolCallProcessor`, but we may
                //       also need to add a new stream detokenizer.
                guard !state.didSendToolCall else {
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
                            try? state.parser.processEOS()
                            if let toolCall = state.toolCall() {
                                state.didSendToolCall = true
                                return .toolCall(toolCall)
                            } else {
                                return nil
                            }
                        }
                    }

                    let messageCount = state.parser.messages.count
                    try state.parser.process(token)

                    if let delta = state.parser.delta {
                        if state.parser.channel == "analysis" {
                            return .reasoning(delta)
                        } else if state.parser.channel == "final" {
                            return .text(delta)
                        } else if state.parser.channel == "commentary",
                                  state.parser.recipient == nil
                        { return .text(delta) }
                    }

                    guard state.hasToolCall(previousMessageCount: messageCount) else {
                        // NOTE: Continue inference because we are
                        //       expect more tokens.
                        return nil
                    }

                    // NOTE: This shouldn't be possible to reach yet because, as mentioned
                    //       above, MLX Swift will not send us the `<|call|>` token because
                    //       we've added it to `extraEOSTokens`. But, once MLX Swift
                    //       supports Harmony natively, we will be able to reach this code.
                    state.didSendToolCall = true
                    if let toolCall = state.toolCall() {
                        try? state.parser.processEOS()
                        return .toolCall(toolCall)
                    } else {
                        // NOTE: Stop inference after seeing any tool call, even
                        //       if it's not valid.
                        try? state.parser.processEOS()
                        return nil
                    }
                } catch {
                    // NOTE: Stop inference after an error
                    try? state.parser.processEOS()
                    return nil
                }
            }
        }
    }

    private struct State {
        var parser = Harmony.StreamableParser(startingRole: .assistant)
        var didSendToolCall = false

        func hasToolCall(previousMessageCount: Int) -> Bool {
            if parser.messages.count > previousMessageCount,
               let lastMessage = parser.messages.last,
               lastMessage.author.role == .assistant,
               let recipient = lastMessage.recipient,
               recipient.hasPrefix("functions.")
            { true }
            else { false }
        }

        mutating func toolCall() -> ToolCall? {
            guard let lastMessage = parser.messages.last,
                  lastMessage.author.role == .assistant,
                  let recipient = lastMessage.recipient,
                  recipient.hasPrefix("functions.")
            else { return nil }

            let functionName = String(recipient.dropFirst("functions.".count))
            let decoder = JSONDecoder()

            guard case let .text(content) = lastMessage.content.first,
                  let jsonData = content.data(using: .utf8),
                  let jsonObject = try? decoder.decode(JSONValue.self, from: jsonData),
                  let args = jsonObject.anyValue as? [String: Any]
            else { return nil }

            let toolCall = ToolCall(
                function: ToolCall.Function(
                    name: functionName,
                    arguments: args
                )
            )
            return toolCall
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
