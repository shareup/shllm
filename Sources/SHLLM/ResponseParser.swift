import Foundation
import class MLXLLM.GPTOSSModel
import class MLXLLM.NemotronHModel
import class MLXLLM.Qwen2Model
import class MLXLLM.Qwen3Model
import class MLXLLM.Qwen3MoEModel
import enum MLXLMCommon.Generation
import struct MLXLMCommon.ToolCall
import class MLXVLM.Gemma4
import class MLXVLM.Gemma4Unified
import class MLXVLM.Mistral3VLM
import class MLXVLM.Qwen35
import class MLXVLM.Qwen35MoE
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

public extension LLM where Model == Gemma4 {
    static var gemma4Parser: ResponseParser { Gemma4ChannelParser<Model>().parser }
}

public extension LLM where Model == Gemma4Unified {
    static var gemma4Parser: ResponseParser { Gemma4ChannelParser<Model>().parser }
}

public extension LLM where Model == Qwen35 {
    static func qwen3_5Parser(for input: UserInput) -> ResponseParser {
        qwen35Parser(for: input)
    }
}

public extension LLM where Model == Qwen35MoE {
    static func qwen3_5MoEParser(for input: UserInput) -> ResponseParser {
        qwen35Parser(for: input)
    }
}

public extension LLM where Model == NemotronHModel {
    static var nemotronParser: ResponseParser {
        ThinkingTagProcessor<NemotronHModel>.defaultsToThinking()
    }
}

public extension LLM where Model == Mistral3VLM {
    static var mistral3Parser: ResponseParser {
        ThinkingTagProcessor<Mistral3VLM>.hybrid(
            startTags: ["[THINK]", "[THINK]\n"],
            endTags: ["[/THINK]", "[/THINK]\n"]
        )
    }

    static var devstral2Parser: ResponseParser { mistral3Parser }
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

    static func qwen35Parser(for input: UserInput) -> ResponseParser {
        let enableThinking = input.additionalContext?["enable_thinking"] as? Bool
        // NOTE: Qwen3.5 models usually default to thinking mode. Only the 2B
        //       models default to non-thinking mode. So, if the `enable_thinking`
        //       flag is not set, we will default to thinking mode.
        if enableThinking == false {
            return hybridParser
        } else {
            return defaultsToThinkingParser
        }
    }
}

// MARK: - Gemma 4 Channel Parser

/// Parses Gemma 4's channel markers to separate reasoning from text.
///
/// Gemma 4 models wrap their output in channels delimited by special tokens:
/// - `<|channel>thought\n` starts a thinking block (reasoning)
/// - `<channel|>` ends the thinking block and starts the final response (text)
///
/// The model always begins with `<|channel>thought\n`, so the parser starts by
/// buffering until the thought-channel header is consumed.  When `<channel|>`
/// is encountered, it switches to text mode.  The channel markers themselves
/// are consumed and not emitted.
///
/// Reference: https://huggingface.co/google/gemma-4-12B-it#channel-thought
final class Gemma4ChannelParser<Model: LanguageModel>: @unchecked Sendable {
    private let state = Locked(ParserState())

    var parser: LLM<Model>.ResponseParser {
        LLM.ResponseParser { (generation: Generation) -> Response? in
            self.state.access { state in
                state.process(generation)
            }
        }
    }

    private struct ParserState {
        private enum Mode {
            case detectingHeader
            case reasoning
            case text
        }

        private var mode = Mode.detectingHeader
        private var buffer = ""
        private var queuedResponses = [Response]()

        mutating func process(_ generation: Generation) -> Response? {
            switch generation {
            case let .chunk(chunk):
                process(chunk)

            case let .toolCall(toolCall):
                queuedResponses.append(.toolCall(toolCall))

            case .info:
                break
            }

            return dequeue()
        }

        private mutating func process(_ chunk: String) {
            guard !chunk.isEmpty else { return }

            switch mode {
            case .detectingHeader:
                buffer += chunk
                processHeaderBuffer()

            case .reasoning:
                buffer += chunk
                processReasoningBuffer()

            case .text:
                enqueueText(chunk)
            }
        }

        private mutating func processHeaderBuffer() {
            let thoughtHeader = Gemma4ChannelMarkers.thoughtHeader
            let endTag = Gemma4ChannelMarkers.endTag

            if buffer.hasPrefix(thoughtHeader) {
                let remainder = String(buffer.dropFirst(thoughtHeader.count))
                buffer = ""
                mode = .reasoning

                if !remainder.isEmpty {
                    buffer = remainder
                    processReasoningBuffer()
                }
                return
            }

            if thoughtHeader.hasPrefix(buffer) {
                return
            }

            if buffer.hasPrefix(endTag) {
                let remainder = String(buffer.dropFirst(endTag.count))
                buffer = ""
                mode = .text
                enqueueText(remainder)
                return
            }

            if endTag.hasPrefix(buffer) {
                return
            }

            let text = buffer
            buffer = ""
            mode = .text
            enqueueText(text)
        }

        private mutating func processReasoningBuffer() {
            let endTag = Gemma4ChannelMarkers.endTag

            if let range = buffer.range(of: endTag) {
                let reasoning = String(buffer[..<range.lowerBound])
                let text = String(buffer[range.upperBound...])
                buffer = ""
                mode = .text
                enqueueReasoning(reasoning)
                enqueueText(text)
                return
            }

            let pendingLength = buffer.lengthOfSuffixMatchingPrefix(of: endTag)
            guard pendingLength > 0 else {
                enqueueReasoning(buffer)
                buffer = ""
                return
            }

            let reasoningEndIndex = buffer.index(buffer.endIndex, offsetBy: -pendingLength)
            let reasoning = String(buffer[..<reasoningEndIndex])
            buffer = String(buffer[reasoningEndIndex...])
            enqueueReasoning(reasoning)
        }

        private mutating func enqueueReasoning(_ reasoning: String) {
            guard !reasoning.isEmpty else { return }
            queuedResponses.append(.reasoning(reasoning))
        }

        private mutating func enqueueText(_ text: String) {
            guard !text.isEmpty else { return }
            queuedResponses.append(.text(text))
        }

        private mutating func dequeue() -> Response? {
            guard !queuedResponses.isEmpty else { return nil }
            return queuedResponses.removeFirst()
        }
    }
}

private enum Gemma4ChannelMarkers {
    static let thoughtHeader = "<|channel>thought\n"
    static let endTag = "<channel|>"
}

private extension String {
    func lengthOfSuffixMatchingPrefix(of marker: String) -> Int {
        let maxLength = Swift.min(count, marker.count - 1)
        guard maxLength > 0 else { return 0 }

        for length in stride(from: maxLength, through: 1, by: -1) {
            let suffixStartIndex = index(endIndex, offsetBy: -length)
            let suffix = String(self[suffixStartIndex...])
            if marker.hasPrefix(suffix) {
                return length
            }
        }

        return 0
    }
}
