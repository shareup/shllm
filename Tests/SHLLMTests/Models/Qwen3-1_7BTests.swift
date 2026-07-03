import Foundation
import MLXLLM
import MLXLMCommon
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Qwen3__1_7BTests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try qwen3__1_7B(input) else { return }

        var reasoning = ""
        var result = ""
        for try await reply in llm {
            switch reply {
            case let .reasoning(text):
                reasoning.append(text)
            case let .text(text):
                result.append(text)
            case .toolCall:
                Issue.record()
            }
        }

        Swift.print("<think>\n\(reasoning)\n</think>")
        #expect(!reasoning.isEmpty)

        Swift.print(result)
        #expect(!result.isEmpty)
    }

    @Test
    func canStreamTextResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try qwen3__1_7B(input) else { return }

        var result = ""
        for try await reply in llm.text {
            result.append(reply)
        }

        Swift.print(result)
        #expect(!result.isEmpty)
    }

    @Test
    func canAwaitResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try qwen3__1_7B(input) else { return }

        let (_reasoning, _text, toolCalls) = try await llm.result

        let reasoning = try #require(_reasoning)
        Swift.print("<think>\n\(reasoning)\n</think>")
        #expect(!reasoning.isEmpty)

        let text = try #require(_text)
        Swift.print(text)
        #expect(!text.isEmpty)

        #expect(toolCalls == nil)
    }

    @Test
    func canAwaitTextResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try qwen3__1_7B(input) else { return }

        let result = try await llm.text.result
        Swift.print(result)
        #expect(!result.isEmpty)
    }

    @Test
    func canFetchTheWeather() async throws {
        let input = UserInput(chat: [
            .system(
                "You are a weather assistant who must use the get_current_weather tool to fetch weather data for any location the user asks about."
            ),
            .user("What is the weather in Paris, France?"),
        ])

        guard let llm = try qwen3__1_7B(
            input,
            tools: [weatherTool]
        ) else { return }

        var reasoning = ""
        var reply = ""
        var toolCallCount = 0
        var weatherLocationFound = false

        for try await response in llm {
            switch response {
            case let .reasoning(text):
                reasoning.append(text)
            case let .text(text):
                reply.append(text)
            case let .toolCall(toolCall):
                toolCallCount += 1
                #expect(toolCall.function.name == "get_current_weather")

                if case let .string(location) = toolCall.function.arguments["location"] {
                    weatherLocationFound = location.lowercased().contains("paris")
                }
            }
        }

        #expect(!reasoning.isEmpty)
        #expect(reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(toolCallCount == 1)
        #expect(weatherLocationFound)
    }

    @Test
    func canUseToolsWithNonStringArgumentsAndRespond() async throws {
        let chat: [Chat.Message] = [
            .system("""
            You are an email assistant. When asked to read an email, call mail_read exactly once.
            After the tool result is provided, reply with the email subject exactly and do not call tools again.
            """),
            .user("Read email 158348 from account me@example.com in mailbox INBOX."),
        ]

        var input = UserInput(chat: chat)

        guard let llm1 = try qwen3__1_7B(
            input,
            tools: [mailReadTool]
        ) else { return }

        let (reasoning, text, toolCallsOpt) = try await llm1.result
        let toolCall = try #require(toolCallsOpt?.first)

        #expect(reasoning != nil)
        #expect(text == nil)
        #expect(toolCall.function.name == "mail_read")
        #expect(toolCall.function.arguments["account"] == .string("me@example.com"))
        #expect(toolCall.function.arguments["mailbox"] == .string("INBOX"))

        let idArgument = try #require(toolCall.function.arguments["id"])
        #expect(idArgument == .int(158_348))
        #expect(idArgument != .string("158348"))

        input.appendAssistantToolCall(toolCall)
        input.appendToolResult(MailReadResponse(subject: mailReadSubject))

        guard let llm2 = try qwen3__1_7B(
            input,
            tools: [mailReadTool]
        ) else { return }

        let (reasoning2, text2, toolCallsOpt2) = try await llm2.result
        let result = try #require(text2)
        Swift.print(result)
        #expect(reasoning2 != nil)
        #expect(result.contains(mailReadSubject))
        #expect(toolCallsOpt2 == nil)
    }
}

private func qwen3__1_7B(
    _ input: UserInput,
    tools: [any ToolProtocol] = []
) throws -> LLM<Qwen3Model>? {
    try loadModel(
        directory: LLM<Qwen3Model>.qwen3__1_7B,
        input: input,
        tools: tools,
        responseParser: LLM<Qwen3Model>.qwen3Parser
    )
}
