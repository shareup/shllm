import Foundation
import MLX
import MLXLMCommon
import Tokenizers

struct TruncatingUserInputProcessor: UserInputProcessor {
    private let baseProcessor: UserInputProcessor
    private let tokenizer: Tokenizer
    private let maxInputTokenCount: Int?

    init(
        wrapping baseProcessor: UserInputProcessor,
        tokenizer: Tokenizer,
        maxInputTokenCount: Int?
    ) {
        self.baseProcessor = baseProcessor
        self.tokenizer = tokenizer
        self.maxInputTokenCount = maxInputTokenCount
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard let maxInputTokenCount else {
            return try await baseProcessor.prepare(input: input)
        }

        var processedInput = input
        switch input.prompt {
        case let .chat(messages):
            let trimmedMessages = try await truncate(
                messages: messages,
                maxTokenCount: maxInputTokenCount
            )
            processedInput.prompt = .chat(trimmedMessages)

        case let .messages(messages):
            let trimmedMessages = try await truncate(
                messages: messages,
                maxTokenCount: maxInputTokenCount
            )
            processedInput.prompt = .messages(trimmedMessages)

        case let .text(prompt):
            let truncatedPrompt = try await truncate(
                text: prompt,
                maxTokenCount: maxInputTokenCount
            )
            processedInput.prompt = .text(truncatedPrompt)
        }

        return try await baseProcessor.prepare(input: processedInput)
    }

    private func truncate(
        messages: [Chat.Message],
        maxTokenCount: Int
    ) async throws -> [Chat.Message] {
        let systemMessages = messages.filter { $0.role == .system }
        let nonSystemMessages = messages.filter { $0.role != .system }

        let systemTokenCount = try await tokenCount(for: systemMessages)
        let remainingTokens = maxTokenCount - systemTokenCount

        guard remainingTokens > 0 else {
            throw SHLLMError.inputTooLong
        }

        let recentMessages = try await recentMessages(
            nonSystemMessages,
            limitedTo: remainingTokens
        )

        return systemMessages + recentMessages
    }

    private func truncate(
        messages: [[String: Any]],
        maxTokenCount: Int
    ) async throws -> [[String: Any]] {
        let systemMessages = messages.filter { ($0["role"] as? String) == "system" }
        let nonSystemMessages = messages.filter { ($0["role"] as? String) != "system" }

        let systemTokenCount = try await tokenCount(for: systemMessages)
        let remainingTokens = maxTokenCount - systemTokenCount

        guard remainingTokens > 0 else {
            throw SHLLMError.inputTooLong
        }

        let recentMessages = try await recentMessages(
            nonSystemMessages,
            limitedTo: remainingTokens
        )

        return systemMessages + recentMessages
    }

    private func truncate(
        text: String,
        maxTokenCount: Int
    ) async throws -> String {
        let tokens = tokenizer.encode(text: text)

        if tokens.count <= maxTokenCount {
            return text
        }

        let half = Double(maxTokenCount) / 2
        let firstHalf = tokens.prefix(Int(half.rounded(.down)))
        let secondHalf = tokens.suffix(Int(half.rounded(.up)))
        assert(firstHalf.count + secondHalf.count <= maxTokenCount)
        let truncatedTokens = Array(firstHalf + secondHalf)
        return tokenizer.decode(tokens: truncatedTokens)
    }

    private func recentMessages(
        _ messages: [Chat.Message],
        limitedTo maxTokenCount: Int
    ) async throws -> [Chat.Message] {
        var selectedMessages: [Chat.Message] = []
        var currentTokenCount = 0

        for message in messages.reversed() {
            let messageTokens = try await tokenCount(for: [message])

            if currentTokenCount + messageTokens <= maxTokenCount {
                selectedMessages.insert(message, at: 0)
                currentTokenCount += messageTokens
            } else {
                break
            }
        }

        return selectedMessages
    }

    private func recentMessages(
        _ messages: [[String: Any]],
        limitedTo maxTokenCount: Int
    ) async throws -> [[String: Any]] {
        var selectedMessages: [[String: Any]] = []
        var currentTokenCount = 0
        for message in messages.reversed() {
            let messageTokens = try await tokenCount(for: [message])
            if currentTokenCount + messageTokens <= maxTokenCount {
                selectedMessages.insert(message, at: 0)
                currentTokenCount += messageTokens
            } else {
                break
            }
        }
        return selectedMessages
    }

    private func tokenCount(for messages: [Chat.Message]) async throws -> Int {
        let combinedContent = messages.map(\.content).joined(separator: "\n")
        let tokens = tokenizer.encode(text: combinedContent)
        return tokens.count
    }

    private func tokenCount(for messages: [[String: Any]]) async throws -> Int {
        var content = messages.reduce(
            into: ""
        ) { (acc: inout String, message: [String: Any]) in
            guard let content = message["content"] as? String,
                  !content.isEmpty
            else { return }
            acc += content + "\n"
        }
        if content.hasSuffix("\n") {
            content.removeLast(1)
        }
        let tokens = tokenizer.encode(text: content)
        return tokens.count
    }
}
