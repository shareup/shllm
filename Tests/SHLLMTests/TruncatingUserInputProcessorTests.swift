import Foundation
import MLX
import MLXLMCommon
@testable import SHLLM
import Testing
import Tokenizers

@Suite(.serialized)
struct TruncatingUserInputProcessorTests {
    @Test
    func textPromptTruncation() async throws {
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
    func chatMessagesPreserveSystem() async throws {
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

        #expect(result.text.tokens.count == 8)
        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 2, 3, 4, 5, 6, 9, 10])
    }

    @Test
    func rawMessagesPreserveSystem() async throws {
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

        #expect(result.text.tokens.count == 8)
        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 2, 3, 4, 5, 6, 9, 10])
    }

    @Test
    func systemMessagesOnlyExceedsLimit() async throws {
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
    func rawSystemMessagesOnlyExceedsLimit() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 2
        )

        let messages: [[String: Any]] = [
            ["role": "system", "content": "1 2 3 4 5 6 7"],
        ]

        let input = UserInput(messages: messages)

        await #expect(throws: SHLLMError.self) {
            try await processor.prepare(input: input)
        }
    }

    @Test
    func noMessageTruncationWhenUnderLimit() async throws {
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
    func noRawMessageTruncationWhenUnderLimit() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 100
        )

        let messages: [[String: Any]] = [
            ["role": "system", "content": "1 2"],
            ["role": "user", "content": "3 4"],
        ]

        let input = UserInput(messages: messages)
        let result = try await processor.prepare(input: input)

        #expect(result.text.tokens.count == 4)
        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 2, 3, 4])
    }

    @Test
    func noLimitPassesThrough() async throws {
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
    func recentMessagesKept() async throws {
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

        #expect(result.text.tokens.count == 5)
        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 2, 3, 4, 7]) // 4 and 7 are combined when flattened
    }

    @Test
    func recentRawMessagesKept() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 6
        )

        let messages: [[String: Any]] = [
            ["role": "system", "content": "1"],
            ["role": "user", "content": "2 3 4"],
            ["role": "assistant", "content": "5 6"],
            ["role": "user", "content": "7"],
        ]

        let input = UserInput(messages: messages)
        let result = try await processor.prepare(input: input)

        #expect(result.text.tokens.count == 5)
        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 2, 3, 4, 7]) // 4 and 7 are combined when flattened
    }

    @Test
    func alternatingAlgorithm() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 50
        )

        let messages: [Chat.Message] = [
            .user("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26"), // 26
            .assistant("27 28 29 30 31 32 33 34 35"), // 9
            .user("36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52"), // 17
            .assistant("53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69"), // 17
            .user("70 71 72 73"), // 4
        ]

        let input = UserInput(chat: messages)
        let result = try await processor.prepare(input: input)

        let rawTokens = result.text.tokens.asArray(Int.self)

        #expect(rawTokens.count == 47)
        #expect(Array(rawTokens[0 ..< 26]) == Array(1 ... 26))
        #expect(Array(rawTokens[26 ..< 43]) == Array(53 ... 69))
        #expect(Array(rawTokens[43 ..< 47]) == Array(70 ... 73))
    }

    @Test
    func messageFlattening() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 7
        )

        let messages: [Chat.Message] = [
            .user("1 2"),
            .user("3 4"),
            .assistant("5 6"),
            .assistant("7 8"),
            .user("9 10"),
        ]

        let input = UserInput(chat: messages)
        let result = try await processor.prepare(input: input)

        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens.count == 6)
        #expect(rawTokens == [1, 2, 7, 8, 9, 10])
    }

    @Test
    func rawMessageFlattening() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 7
        )

        let messages: [Message] = [
            ["role": "user", "content": "1 2"],
            ["role": "user", "content": "3 4"],
            ["role": "assistant", "content": "5 6"],
            ["role": "assistant", "content": "7 8"],
            ["role": "user", "content": "9 10"],
        ]

        let input = UserInput(messages: messages)
        let result = try await processor.prepare(input: input)

        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens.count == 6)
        #expect(rawTokens == [1, 2, 7, 8, 9, 10])
    }

    @Test
    func emptyMessages() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 10
        )

        let input = UserInput(chat: [])
        let result = try await processor.prepare(input: input)

        #expect(result.text.tokens.count == 0)
    }

    @Test
    func singleMessage() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 10
        )

        let messages: [Chat.Message] = [
            .user("1 2 3 4 5"),
        ]

        let input = UserInput(chat: messages)
        let result = try await processor.prepare(input: input)

        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 2, 3, 4, 5])
    }

    @Test
    func exactTokenLimit() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 5
        )

        let messages: [Chat.Message] = [
            .user("1 2 3 4 5"),
        ]

        let input = UserInput(chat: messages)
        let result = try await processor.prepare(input: input)

        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens.count == 5)
        #expect(rawTokens == [1, 2, 3, 4, 5])
    }

    @Test
    func singleMessageExceedsTokenLimit() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 4
        )

        let messages: [Chat.Message] = [
            .user("1 2 3 4 5"),
        ]

        let input = UserInput(chat: messages)
        let result = try await processor.prepare(input: input)

        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens.isEmpty)
    }

    @Test
    func singleRawMessageExceedsTokenLimit() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 4
        )

        let messages: [[String: Any]] = [
            ["role": "user", "content": "1 2 3 4 5"],
        ]

        let input = UserInput(messages: messages)
        let result = try await processor.prepare(input: input)

        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens.isEmpty)
    }

    @Test
    func onlySystemMessages() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 10
        )

        let messages: [Chat.Message] = [
            .system("1 2 3 4 5"),
        ]

        let input = UserInput(chat: messages)
        let result = try await processor.prepare(input: input)

        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens == [1, 2, 3, 4, 5])
    }

    @Test
    func multipleSystemMessages() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 10
        )

        let messages: [Chat.Message] = [
            .system("1 2"),
            .system("3 4"),
            .user("5 6 7"),
            .assistant("8 9 10 11"),
        ]

        let input = UserInput(chat: messages)
        let result = try await processor.prepare(input: input)

        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens.count == 8)
        #expect(rawTokens == [1, 2, 3, 4, 8, 9, 10, 11])
    }

    @Test
    func veryLongFirstMessage() async throws {
        let tokenizer = NaiveTokenizer()
        let baseProcessor = NaiveInputProcessor()
        let processor = TruncatingUserInputProcessor(
            wrapping: baseProcessor,
            tokenizer: tokenizer,
            maxInputTokenCount: 15
        )

        let messages: [Chat.Message] = [
            .system("1 2"),
            .user("3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18"),
            .assistant("19 20"),
            .user("21 22"),
        ]

        let input = UserInput(chat: messages)
        let result = try await processor.prepare(input: input)

        let rawTokens = result.text.tokens.asArray(Int.self)
        #expect(rawTokens.count == 6)
        #expect(rawTokens == [1, 2, 19, 20, 21, 22])
    }
}

private final class NaiveTokenizer: Tokenizer {
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

    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int] {
        encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }

    var bosToken: String? { nil }
    var bosTokenId: Int? { nil }
    var eosToken: String? { nil }
    var eosTokenId: Int? { nil }
    var unknownToken: String? { nil }
    var unknownTokenId: Int? { nil }
    var hasChatTemplate: Bool { true }

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
        additionalContext _: [String: any Sendable]?
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
        additionalContext _: [String: any Sendable]?
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
