import Foundation
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Qwen2_5__7BTests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try qwen2_5__7B(input) else { return }

        var result = ""
        for try await reply in llm {
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

        guard let llm = try qwen2_5__7B(input) else { return }

        let result = try await llm.result
        Swift.print(result)
        #expect(!result.isEmpty)
    }

    @Test
    func canFetchTheWeather() async throws {
        do {
            let input: UserInput = .init(
                messages: [
                    [
                        "role": "system",
                        "content": "You are a weather fetching assistant. Your only purpose is to fetch weather data.",
                    ],
                    ["role": "system", "content": "The user prefers F°."],
                    ["role": "user", "content": "What is weather in Paris like?"],
                ],
                tools: Tools([weatherToolFunction]).toSpec()
            )

            guard let llm = try qwen2_5__7B(input) else { return }

            let tool: WeatherTool = try await llm.toolResult()
            let expectedTool = WeatherTool.getCurrentWeather(.init(
                location: "Paris, France",
                unit: .fahrenheit
            ))

            print("\(#function) 1:", tool)
            #expect(tool == expectedTool)
        }

        do {
            let input: UserInput = .init(
                messages: [
                    [
                        "role": "system",
                        "content": "You are weather fetching assistant. Your only purpose is to fetch weather data.",
                    ],
                    ["role": "system", "content": "The user prefers C°."],
                    ["role": "user", "content": "What is weather in Paris like?"],
                ],
                tools: Tools([weatherToolFunction]).toSpec()
            )

            guard let llm = try qwen2_5__7B(input) else { return }

            let tool: WeatherTool = try await llm.toolResult()
            let expectedTool = WeatherTool.getCurrentWeather(.init(
                location: "Paris, France",
                unit: .celsius
            ))

            print("\(#function) 2:", tool)
            #expect(tool == expectedTool)
        }
    }
}

private func qwen2_5__7B(
    _ input: UserInput
) throws -> LLM<Qwen2Configuration, Qwen2Model>? {
    try loadModel(
        LLM.qwen2_5__7B,
        directory: LLM.qwen2_5__7B,
        input: input
    )
}
