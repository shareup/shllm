import Foundation
import class MLXLLM.GPTOSSModel
import class MLXLLM.NemotronHModel
import class MLXLLM.Qwen2Model
import class MLXLLM.Qwen3Model
import class MLXLLM.Qwen3MoEModel
import enum MLXLMCommon.Generation
import struct MLXLMCommon.ToolCall
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

public extension LLM where Model == NemotronHModel {
    static var nemotronParser: ResponseParser {
        let state = Locked(NemotronParserState())
        return ResponseParser { (generation: Generation) -> Response? in
            state.access { state in
                state.process(generation)
            }
        }
    }
}

private struct NemotronParserState {
    private let thinkStartTags: Set<String> = ["<think>", "<think>\n"]
    private let thinkEndTags: Set<String> = ["</think>", "</think>\n"]
    private let toolCallStartTag = "<tool_call>"
    private let toolCallEndTag = "</tool_call>"

    private var isThinking = true
    private var isBufferingToolCall = false
    private var textBuffer = ""

    mutating func process(_ generation: Generation) -> Response? {
        switch generation {
        case let .chunk(chunk):
            return processChunk(chunk)
        case let .toolCall(toolCall):
            isThinking = false
            isBufferingToolCall = false
            textBuffer = ""
            return .toolCall(toolCall)
        case .info:
            return finalize()
        }
    }

    private mutating func processChunk(_ chunk: String) -> Response? {
        if thinkStartTags.contains(chunk) {
            isThinking = true
            return nil
        }

        if thinkEndTags.contains(chunk) {
            isThinking = false
            return nil
        }

        if isThinking {
            return .reasoning(chunk)
        }

        textBuffer.append(chunk)

        if isBufferingToolCall {
            if textBuffer.contains(toolCallEndTag) {
                let result = parseToolCall()
                textBuffer = ""
                isBufferingToolCall = false
                return result
            }
            return nil
        }

        if textBuffer.contains(toolCallStartTag) {
            isBufferingToolCall = true
            return nil
        }

        let trimmed = textBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty, toolCallStartTag.hasPrefix(trimmed) {
            return nil
        }

        let text = textBuffer
        textBuffer = ""
        return .text(text)
    }

    private mutating func finalize() -> Response? {
        guard !textBuffer.isEmpty else { return nil }

        if let result = parseToolCall() {
            textBuffer = ""
            return result
        }

        let text = textBuffer
        textBuffer = ""
        return text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? nil : .text(text)
    }

    private func parseToolCall() -> Response? {
        guard let funcStart = textBuffer.range(of: "<function="),
              let funcNameEnd = textBuffer.range(
                  of: ">",
                  range: funcStart.upperBound ..< textBuffer.endIndex
              ),
              let funcEnd = textBuffer.range(of: "</function>")
        else { return nil }

        let funcName = String(textBuffer[funcStart.upperBound ..< funcNameEnd.lowerBound])
        let paramSection = String(textBuffer[funcNameEnd.upperBound ..< funcEnd.lowerBound])

        var arguments: [String: any Sendable] = [:]
        var searchRange = paramSection.startIndex ..< paramSection.endIndex

        while let paramStart = paramSection.range(
            of: "<parameter=", range: searchRange
        ) {
            guard let nameEnd = paramSection.range(
                of: ">", range: paramStart.upperBound ..< paramSection.endIndex
            ) else { break }

            let paramName = String(paramSection[paramStart.upperBound ..< nameEnd.lowerBound])

            guard let paramEnd = paramSection.range(
                of: "</parameter>", range: nameEnd.upperBound ..< paramSection.endIndex
            ) else { break }

            var value = String(paramSection[nameEnd.upperBound ..< paramEnd.lowerBound])
            if value.hasPrefix("\n") { value = String(value.dropFirst()) }
            if value.hasSuffix("\n") { value = String(value.dropLast()) }

            arguments[paramName] = value
            searchRange = paramEnd.upperBound ..< paramSection.endIndex
        }

        return .toolCall(ToolCall(
            function: .init(name: funcName, arguments: arguments)
        ))
    }
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

private extension LLM {
    static var hybridParser: ResponseParser {
        ThinkingTagProcessor<Model>.hybrid()
    }

    static var defaultsToThinkingParser: ResponseParser {
        ThinkingTagProcessor<Model>.defaultsToThinking()
    }
}
