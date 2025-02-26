import Foundation
@testable import SHLLM
import Testing

extension DeepSeekR1: InitializableWithDirectory {
    static var tests: Self? {
        get async throws {
            try await loadModel(from: bundleDirectory)
        }
    }
}

@Test
func canLoadAndQueryDeepSeekR1() async throws {
    guard let llm = try await DeepSeekR1.tests else { return }
    let result = try await llm.request(.init(messages: [
        ["role": "system", "content": "You are a helpful assistant."],
        ["role": "user", "content": "What is the meaning of life?"],
    ]))
    Swift.print(result)
    #expect(!result.isEmpty)
}

@Test
func canHelpMeFetchTheWeatherWithR1() async throws {
    guard let llm = try await DeepSeekR1.tests else { return }

    let tools = Tools([
        .init(
            name: "get_current_weather",
            description: "Get the current weather in a given location",
            parameters: [
                .string(
                    name: "location",
                    description: "The city and state, e.g. San Francisco, CA",
                    required: true
                ),
                .string(name: "unit", restrictTo: ["celsius", "fahrenheit"]),
            ]
        ),
    ])

    let tool1: WeatherTool = try await llm.request(
        tools: tools,
        messages: [
            [
                "role": "system",
                "content": "You are a weather fetching assistant. Your only purpose is to fetch weather data.",
            ],
            ["role": "user", "content": "What is weather in Paris like?"],
        ]
    )

    let expectedTool1 = WeatherTool.getCurrentWeather(
        location: "Paris, France",
        unit: .fahrenheit
    )

    #expect(tool1 == expectedTool1)

    let tool2: WeatherTool = try await llm.request(
        tools: tools,
        messages: [
            [
                "role": "system",
                "content": "You are weather fetching assistant. Your only purpose is to fetch weather data.",
            ],
            ["role": "system", "content": "The user prefers C°."],
            ["role": "user", "content": "What is weather in Paris like?"],
        ]
    )

    let expectedTool2 = WeatherTool.getCurrentWeather(
        location: "Paris, France",
        unit: .celsius
    )

    #expect(tool2 == expectedTool2)
}
