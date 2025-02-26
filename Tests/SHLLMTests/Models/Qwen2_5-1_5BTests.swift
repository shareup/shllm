@testable import SHLLM
import Testing

private extension Qwen2_5__1_5B {
    init() async throws {
        try await self.init(directory: Self.bundleDirectory)
    }
}

@Test
func canLoadAndQueryQwen2_5__1_5B() async throws {
    let llm = try await Qwen2_5__1_5B()
    let result = try await llm.request(.init(messages: [
        ["role": "system", "content": "You are a helpful assistant."],
        ["role": "user", "content": "What is the meaning of life?"],
    ]))
    Swift.print(result)
    #expect(!result.isEmpty)
}

@Test
func canHelpMeFetchTheWeather() async throws {
    let llm = try await Qwen2_5__1_5B()

    let toolSpec: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": [
                "type": "object",
                "properties": [
                    "location": [
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    ] as [String: String],
                    "unit": [
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    ] as [String: any Sendable],
                ] as [String: [String: any Sendable]],
                "required": ["location"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ] as [String: any Sendable]

    let result = try await llm.request(.init(
        messages: [
            ["role": "system", "content": "You are weather fetching assistant. Your only purpose is to fetch weather data."],
            ["role": "user", "content": "What is weather in Paris like?"],
        ],
        tools: [toolSpec]
    ))

    let expectedResult = """
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Paris, France", "unit": "fahrenheit"}}
</tool_call>
""".trimmingCharacters(in: .whitespacesAndNewlines)

    #expect(result == expectedResult)

    let result2 = try await llm.request(.init(
        messages: [
            ["role": "system", "content": "You are weather fetching assistant. Your only purpose is to fetch weather data."],
            ["role": "system", "content": "The user prefers Cº."],
            ["role": "user", "content": "What is weather in Paris like?"],
        ],
        tools: [toolSpec]
    ))

    let expectedResult2 = """
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Paris, France", "unit": "celsius"}}
</tool_call>
""".trimmingCharacters(in: .whitespacesAndNewlines)

    #expect(result2 == expectedResult2)
}
