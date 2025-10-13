import Foundation
import MLXLLM
import MLXLMCommon
@testable import SHLLM
import Testing

@Suite(.serialized)
struct GPTOSS_20BTests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try gptOSS_20B(input) else { return }

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

        guard let llm = try gptOSS_20B(input) else { return }

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

        guard let llm = try gptOSS_20B(
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
            case .reasoning:
                break
            }
        }

        #expect(!reply.isEmpty)
        #expect(toolCallCount >= 1)
        #expect(weatherLocationFound)
    }

    @Test
    func canSeparateReasoningFromFinalOutput() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try gptOSS_20B(input) else { return }

        var reasoning = ""
        var finalText = ""

        for try await response in llm {
            Swift.print("$$$", response)
            switch response {
            case let .reasoning(content):
                reasoning += content
            case let .text(content):
                finalText += content
            case .toolCall:
                break
            }
        }

        Swift.print("Reasoning: \(reasoning)")
        Swift.print("Final text: \(finalText)")

        #expect(!reasoning.isEmpty)
        #expect(!finalText.isEmpty)
    }

//    @Test
//    func canStreamReasoningOnly() async throws {
//        let input: UserInput = .init(messages: [
//            ["role": "system", "content": "You are a helpful assistant."],
//            ["role": "user", "content": "What is the meaning of life?"],
//        ])
//
//        guard let llm = try gptOSS_20B(input) else { return }
//
//        var reasoning = ""
//        for try await chunk in llm.reasoning {
//            reasoning += chunk
//        }
//
//        Swift.print("Reasoning only: \(reasoning)")
//        #expect(!reasoning.isEmpty)
//    }

    @Test
    func canStreamTextOnly() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try gptOSS_20B(input) else { return }

        var text = ""
        for try await chunk in llm.text {
            text += chunk
        }

        Swift.print("Text only: \(text)")
        #expect(!text.isEmpty)
    }
}

private func gptOSS_20B(
    _ input: UserInput,
    tools: [any ToolProtocol] = []
) throws -> LLM<GPTOSSModel>? {
    try loadModel(
        directory: LLM<GPTOSSModel>.gptOSS_20B_8bit,
        input: input,
        tools: tools
    )
}
