import Foundation
import class MLXLLM.GPTOSSModel
import class MLXLLM.LFM2MoEModel
import class MLXLLM.Qwen2Model
import class MLXLLM.Qwen3Model
import class MLXLLM.Qwen3MoEModel
import enum MLXLMCommon.Generation
import class MLXVLM.Qwen3VL
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
    static var deepSeekR1Parser = defaultsToThinkingParser
}

public extension LLM where Model == Qwen3Model {
    static var qwen3Parser = hybridParser
}

public extension LLM where Model == Qwen3MoEModel {
    static var qwen3MoEParser = hybridParser
}

public extension LLM where Model == Qwen3VL {
    static var qwen3VLInstructParser = defaultParser
    static var qwen3VLThinkingParser = defaultsToThinkingParser
}

public extension LLM where Model == GPTOSSModel {
    static var gptOSSParser: ResponseParser {
        let state = Locked(GPTOSSState())
        return ResponseParser { (generation: Generation) -> Response? in
            state.access { state -> Response? in
                // NOTE: Because MLX Swift does not natively support Harmony,
                //       the tool calls produced by GPT-OSS are not sent as
                //       `Generation.toolCall` and the `<|call|>` token is not
                //       recognized as a stop token. In order to make tool call
                //       work in SHLLM, we manually parse the Harmony message
                //       format and extract the tool call from the message.
                //       However, since it's incorrect to continue inference
                //       after the model produces `<|call|>`, we added that
                //       token to `extraEOSTokens`, which means that MLX Swift
                //       will stop generating tokens when it encounters `<|call|>`.
                //       So, our Harmony parser will never actually see the tool
                //       call token, which means we won't know when to send a
                //       tool call. To work around this, we check for the
                //       presence of a tool call after MLX Swift stops generating
                //       tokens. If one exists, we send it to the client. But, to
                //       prevent a loop where we send the same tool call over and
                //       over again, we need to break if we've already sent the
                //       tool call.
                //
                //       The fix for this will be to add Harmony support directly
                //       to MLX Swift. At the very least, we'll need to add a new
                //       `ToolCallProcessor`, but we may also need to add a new
                //       stream detokenizer.
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
                            } else { return nil }
                        }
                    }

                    let messageCount = state.parser.messages.count
                    try state.parser.process(token)

                    if let delta = state.parser.delta {
                        if state.parser.channel == "analysis" {
                            return .reasoning(delta)
                        } else if state.parser.channel == "final" {
                            return .text(delta)
                        } else if state.parser.channel == "commentary" {
                            if let recipient = state.parser.recipient,
                               recipient.hasPrefix("functions.")
                            {
                                // NOTE: Waiting for tool call materialization
                            } else {
                                return .text(delta)
                            }
                        }
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

    private struct GPTOSSState {
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

public extension LLM where Model == LFM2MoEModel {
    /// Parser for LFM2 which emits tool calls as a Python-style list of
    /// function invocations between special tokens:
    ///   <|tool_call_start|> [func(arg="value"), ...] <|tool_call_end|>
    /// This parser accumulates tokens between the start/end markers,
    /// parses one or more calls, and yields them as `Response.toolCall`.
    ///
    /// # Example Conversation with Tool Call
    ///
    /// ```
    /// <|startoftext|><|im_start|>system
    /// List of tools: <|tool_list_start|>[{"name": "get_candidate_status", "description":
    /// "Retrieves the current status of a candidate in the recruitment process", "parameters":
    /// {"type": "object", "properties": {"candidate_id": {"type": "string", "description":
    /// "Unique identifier for the candidate"}}, "required":
    /// ["candidate_id"]}}]<|tool_list_end|><|im_end|>
    /// <|im_start|>user
    /// What is the current status of candidate ID 12345?<|im_end|>
    /// <|im_start|>assistant
    /// <|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking
    /// the current status of candidate ID 12345.<|im_end|>
    /// <|im_start|>tool
    /// <|tool_response_start|>{"candidate_id": "12345", "status": "Interview Scheduled",
    /// "position": "Clinical Research Associate", "date":
    /// "2023-11-20"}<|tool_response_end|><|im_end|>
    /// <|im_start|>assistant
    /// The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the
    /// position of Clinical Research Associate, with an interview date set for
    /// 2023-11-20.<|im_end|>
    /// ```
    ///
    static var lfm2Parser: ResponseParser {
        let state = Locked(LFM2State.streaming)
        return ResponseParser { (generation: Generation) -> Response? in
            state.access { $0.process(generation) }
        }
    }
}

private enum LFM2State {
    case parsingToolCall(String)
    case streaming

    mutating func process(_ generation: Generation) -> Response? {
        let startToken = "<|tool_call_start|>"
        let endToken = "<|tool_call_end|>"

        switch (self, generation) {
        case (.streaming, .chunk(startToken)):
            self = .parsingToolCall("")
            return nil

        case (.parsingToolCall(var buffer), .chunk(endToken)):
            if let call = Python.parseFunctionCall(&buffer) {
                self = .parsingToolCall(buffer)
                return .toolCall(call)
            } else {
                self = .streaming
                return nil
            }

        case (.parsingToolCall(var buffer), let .chunk(token)):
            buffer += token
            if let call = Python.parseFunctionCall(&buffer) {
                self = .parsingToolCall(buffer)
                return .toolCall(call)
            } else {
                self = .parsingToolCall(buffer)
                return nil
            }

        case let (.streaming, .chunk(token)):
            return .text(token)

        case var (.parsingToolCall(buffer), .info):
            assert(buffer.isEmpty)
            if let call = Python.parseFunctionCall(&buffer) {
                self = .parsingToolCall(buffer)
                return .toolCall(call)
            } else {
                self = .streaming
                return nil
            }

        case (_, .info):
            return nil

        case let (_, .toolCall(call)):
            return .toolCall(call)
        }
    }
}

private extension LLM {
    static var hybridParser: ResponseParser {
        let isThinking = Locked(false)
        let start = Set(["<think>", "<think>\n"])
        let end = Set(["</think>", "</think>\n"])
        return ResponseParser { (generation: Generation) -> Response? in
            isThinking.access { isThinking -> Response? in
                switch generation {
                case let .chunk(chunk):
                    if start.contains(chunk) {
                        isThinking = true
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
                    isThinking = false
                    return .toolCall(toolCall)

                case .info:
                    return nil
                }
            }
        }
    }

    static var defaultsToThinkingParser: ResponseParser {
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
                    isThinking = false
                    return .toolCall(toolCall)

                case .info:
                    return nil
                }
            }
        }
    }
}
