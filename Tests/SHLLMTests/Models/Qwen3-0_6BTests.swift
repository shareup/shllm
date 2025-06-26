import Foundation
import MLXLLM
import MLXLMCommon
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Qwen3__0_6BTests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try qwen3__0_6B(input) else { return }

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

        guard let llm = try qwen3__0_6B(input) else { return }

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

        guard let llm = try qwen3__0_6B(
            input,
            tools: [weatherTool]
        ) else { return }

        var reply = ""
        var toolCallCount = 0
        var weatherLocationFound = false

        for try await response in llm {
            switch response {
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

        #expect(!reply.isEmpty)
        // Qwen 3 0.6B often calls the tool multiple times. I'm not sure why.
        #expect(toolCallCount >= 1)
        #expect(weatherLocationFound)
    }
}

private func qwen3__0_6B(
    _ input: UserInput,
    tools: [any ToolProtocol] = []
) throws -> LLM<Qwen3Model>? {
    try loadModel(
        directory: LLM<Qwen3Model>.qwen3__0_6B,
        input: input,
        tools: tools
    )
}
