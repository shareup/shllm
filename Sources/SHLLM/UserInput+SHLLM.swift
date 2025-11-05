import Foundation
import MLXLMCommon

public extension UserInput {
    /// Append a user message
    mutating func appendUser(_ content: String, images: [UserInput.Image] = []) {
        ensureChatForm()
        appendChatMessage(.user(content, images: images))
    }

    /// Append an assistant message
    mutating func appendAssistant(_ content: String, images: [UserInput.Image] = []) {
        ensureChatForm()
        appendChatMessage(.assistant(content, images: images))
    }

    /// Append a generic tool-result message suitable for most models.
    mutating func appendToolResult(_ object: [String: Any]) {
        ensureMessagesForm()
        let message: Message = [
            "role": "tool",
            "content": object,
        ]
        appendMessage(message)
    }

    /// Append a generic tool-result message suitable for most models.
    mutating func appendToolResult(_ payload: some Encodable) {
        let object = (try? encodeToJSONObject(payload)) ?? [:]
        appendToolResult(object)
    }
}

/// Harmony-specific helpers for building UserInput messages for GPT-OSS.
public extension UserInput {
    /// Append an assistant tool-call message for Harmony/GPT-OSS.
    /// - Parameters:
    ///   - call: The parsed tool call.
    ///   - contentType: Optional content type (defaults to "json" if omitted by the template).
    mutating func appendHarmonyAssistantToolCall(
        _ call: ToolCall,
        contentType: String? = nil
    ) {
        ensureMessagesForm()
        var function: [String: Any] = [
            "name": call.function.name,
            "arguments": call.function.arguments.mapValues { $0.anyValue },
        ]
        if let contentType { function["content_type"] = contentType }

        let message: Message = [
            "role": "assistant",
            "tool_calls": [[
                "type": "function",
                "function": function,
            ]],
        ]

        appendMessage(message)
    }

    /// Append a tool-result message suitable for Harmony/GPT-OSS models.
    mutating func appendHarmonyToolResult(_ object: [String: Any]) {
        ensureMessagesForm()
        let message: Message = [
            "role": "tool",
            "content": object,
        ]
        appendMessage(message)
    }

    /// Append a tool-result message suitable for Harmony/GPT-OSS models.
    mutating func appendHarmonyToolResult(_ payload: some Encodable) {
        let object = (try? encodeToJSONObject(payload)) ?? [:]
        appendHarmonyToolResult(object)
    }

    private mutating func ensureChatForm() {
        switch prompt {
        case .chat:
            return
        case let .messages(messages):
            // Can't convert from messages back to chat
            assertionFailure("Cannot append chat messages after using messages form")
        case let .text(text):
            prompt = .chat([.user(text)])
        }
    }

    private mutating func appendChatMessage(_ message: Chat.Message) {
        switch prompt {
        case var .chat(messages):
            messages.append(message)
            prompt = .chat(messages)
        case .messages, .text:
            assertionFailure("ensureChatForm() must be called before appending")
        }
    }

    private mutating func ensureMessagesForm() {
        switch prompt {
        case .messages:
            return
        case let .chat(chatMessages):
            let raw = DefaultMessageGenerator().generate(messages: chatMessages)
            prompt = .messages(raw)
        case let .text(text):
            let raw = DefaultMessageGenerator().generate(messages: [.user(text)])
            prompt = .messages(raw)
        }
    }

    private mutating func appendMessage(_ message: Message) {
        switch prompt {
        case var .messages(messages):
            messages.append(message)
            prompt = .messages(messages)
        case .chat, .text:
            assertionFailure("ensureMessagesForm() must be called before appending")
        }
    }
}

private func encodeToJSONObject(_ value: some Encodable) throws -> [String: Any]? {
    let data = try JSONEncoder().encode(value)
    return try JSONSerialization.jsonObject(
        with: data,
        options: [.fragmentsAllowed]
    ) as? [String: Any]
}
