@testable import SHLLM
import Testing

extension Phi3: InitializableWithDirectory {
    static var tests: Self? {
        get async throws {
            try await loadModel(from: bundleDirectory)
        }
    }
}

@Test
func canLoadAndQueryPhi3() async throws {
    guard let llm = try await Phi3.tests else { return }
    let result = try await llm.request(.init(messages: [
        ["role": "system", "content": "You are a helpful assistant."],
        ["role": "user", "content": "What is the meaning of life?"],
    ]))
    Swift.print(result)
    #expect(!result.isEmpty)
}

@Test
func canHelpMeFetchTheWeatherWithPhi3() async throws {
    guard let llm = try await Phi3.tests else { return }

    let tool1: WeatherTool = try await llm.request(
        tools: Tools([weatherToolFunction]),
        messages: [
            [
                "role": "system",
                "content": "You are a weather fetching assistant. Your only purpose is to fetch weather data.",
            ],
            ["role": "user", "content": "What is weather in Paris like?"],
        ]
    )

    let expectedTool1 = WeatherTool.getCurrentWeather(.init(
        location: "Paris, France",
        unit: .fahrenheit
    ))

    print("\(#function) 1:", tool1)
    #expect(tool1 == expectedTool1)

    let tool2: WeatherTool = try await llm.request(
        tools: Tools([weatherToolFunction]),
        messages: [
            [
                "role": "system",
                "content": "You are weather fetching assistant. Your only purpose is to fetch weather data.",
            ],
            ["role": "system", "content": "The user prefers C°."],
            ["role": "user", "content": "What is weather in Paris like?"],
        ]
    )

    let expectedTool2 = WeatherTool.getCurrentWeather(.init(
        location: "Paris, France",
        unit: .celsius
    ))

    print("\(#function) 2:", tool2)
    #expect(tool2 == expectedTool2)
}
