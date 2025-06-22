import Foundation
import MLX
import MLXLMCommon
@testable import SHLLM
import Testing
import Tokenizers

@Suite(.serialized)
struct TruncatingUserInputProcessorTests {
    @Test
    func testTextPromptTruncation() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 3
        )

        let longText = "1 2 3 4 5 6"

        let input = UserInput(prompt: .text(longText))
        let result = try await processor.prepare(input: input)

        #expect(result.text.tokens.count == 3)
        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 5, 6])
    }

    @Test
    func testChatMessagesPreserveSystem() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 8
        )

        let messages: [Chat.Message] = [
            .system("1 2 3"),
            .user("4 5 6"),
            .assistant("7 8"),
            .user("9 10"),
        ]

        let input = UserInput(chat: messages)
        let result = try await processor.prepare(input: input)

        #expect(result.text.tokens.count == 7)
        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 2, 3, 7, 8, 9, 10])
    }

    @Test
    func testRawMessagesPreserveSystem() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 8
        )

        let messages: [Message] = [
            ["role": "system", "content": "1 2 3"],
            ["role": "user", "content": "4 5 6"],
            ["role": "assistant", "content": "7 8"],
            ["role": "user", "content": "9 10"],
        ]

        let input = UserInput(messages: messages)
        let result = try await processor.prepare(input: input)

        #expect(result.text.tokens.count == 7)
        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 2, 3, 7, 8, 9, 10])
    }

    @Test
    func testSystemMessagesOnlyExceedsLimit() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 2
        )

        let messages: [Chat.Message] = [
            .system("1 2 3 4 5 6 7"),
        ]

        let input = UserInput(chat: messages)

        await #expect(throws: SHLLMError.self) {
            try await processor.prepare(input: input)
        }
    }

    @Test
    func testNoTruncationWhenUnderLimit() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 100
        )

        let messages: [Chat.Message] = [
            .system("1 2"),
            .user("3 4"),
        ]

        let input = UserInput(chat: messages)
        let result = try await processor.prepare(input: input)

        #expect(result.text.tokens.count == 4)
        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 2, 3, 4])
    }

    @Test
    func testNoLimitPassesThrough() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: nil
        )

        let longText = "1 2 3 4 5 6 7 8"

        let input = UserInput(prompt: .text(longText))
        let result = try await processor.prepare(input: input)

        #expect(result.text.tokens.count == 8)
    }

    @Test
    func testRecentMessagesKept() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 6
        )

        let messages: [Chat.Message] = [
            .system("1"),
            .user("2 3 4"),
            .assistant("5 6"),
            .user("7"),
        ]

        let input = UserInput(chat: messages)
        let result = try await processor.prepare(input: input)

        #expect(result.text.tokens.count == 4)
        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 2, 3, 47]) // 4 and 7 are combined when flattened
    }
}

private class NaiveTokenizer: Tokenizer {
    func tokenize(text: String) -> [String] {
        text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
    }

    func encode(text: String) -> [Int] {
        text.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .enumerated()
            .map { index, word in
                guard let int = Int(word) else {
                    return index + 1
                }
                return int
            }
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        encode(text: text)
    }

    func decode(tokens: [Int]) -> String {
        decode(tokens: tokens, skipSpecialTokens: false)
    }

    func decode(tokens: [Int], skipSpecialTokens _: Bool) -> String {
        tokens
            .map(String.init)
            .joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        Int(token) ?? token.hashValue
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(id)
    }

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] {
        let combined = messages
            .compactMap { $0["content"] as? String }
            .joined(separator: " ")
        return encode(text: combined)
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message],
        tools _: [Tokenizers.ToolSpec]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message],
        tools _: [Tokenizers.ToolSpec]?,
        additionalContext _: [String: Any]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate _: Tokenizers.ChatTemplateArgument
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate _: String
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate _: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt _: Bool,
        truncation _: Bool,
        maxLength _: Int?,
        tools _: [Tokenizers.ToolSpec]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message],
        chatTemplate _: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt _: Bool,
        truncation _: Bool,
        maxLength _: Int?,
        tools _: [Tokenizers.ToolSpec]?,
        additionalContext _: [String: Any]?
    ) throws -> [Int] {
        try applyChatTemplate(messages: messages)
    }
}

private struct NaiveInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        switch input.prompt {
        case let .chat(messages):
            let tokens = messages
                .map(\.content)
                .joined(separator: " ")
                .components(separatedBy: .whitespacesAndNewlines)
                .filter { !$0.isEmpty }
                .enumerated()
                .map { index, word in
                    guard let int = Int(word) else {
                        return index + 1
                    }
                    return int
                }
            return LMInput(text: .init(tokens: MLXArray(tokens)))

        case let .messages(messages):
            let tokens = messages
                .compactMap { $0["content"] as? String }
                .joined(separator: " ")
                .components(separatedBy: .whitespacesAndNewlines)
                .filter { !$0.isEmpty }
                .enumerated()
                .map { index, word in
                    guard let int = Int(word) else {
                        return index + 1
                    }
                    return int
                }
            return LMInput(text: .init(tokens: MLXArray(tokens)))

        case let .text(text):
            let tokens = text
                .components(separatedBy: .whitespacesAndNewlines)
                .filter { !$0.isEmpty }
                .enumerated()
                .map { index, word in
                    guard let int = Int(word) else {
                        return index + 1
                    }
                    return int
                }
            return LMInput(text: .init(tokens: MLXArray(tokens)))
        }
    }
}
