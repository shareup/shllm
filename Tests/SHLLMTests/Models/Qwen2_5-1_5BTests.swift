import Foundation
@testable import SHLLM
import Testing

extension Qwen2_5__1_5B: InitializableWithDirectory {
    static var tests: Self? {
        get async throws {
            try await loadModel(from: bundleDirectory)
        }
    }
}

@Test
func canLoadAndQueryQwen2_5__1_5B() async throws {
    guard let llm = try await Qwen2_5__1_5B.tests else { return }
    let result = try await llm.request(.init(messages: [
        ["role": "system", "content": "You are a helpful assistant."],
        ["role": "user", "content": "What is the meaning of life?"],
    ]))
    Swift.print(result)
    #expect(!result.isEmpty)
}

@Test
func canHelpMeFetchTheWeather() async throws {
    guard let llm = try await Qwen2_5__1_5B.tests else { return }

    let decoder = JSONDecoder()

    struct Tool: Codable, Hashable {
        let name: String
        let arguments: [String: String]
    }

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
                        "description": "The city and state abbreviation ('San Francisco, CA') when a U.S. city OR the city and country abbreviation ('Hamburg, DE') when not a U.S. city.",
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

    let result1 = try await llm.request(.init(
        messages: [
            [
                "role": "system",
                "content": "You are weather fetching assistant. Your only purpose is to fetch weather data.",
            ],
            ["role": "user", "content": "What is weather in Paris like?"],
        ],
        tools: [toolSpec]
    )).trimmingCharacters(in: .whitespacesAndNewlines)

    #expect(result1.hasPrefix("<tool_call>\n"))
    #expect(result1.hasSuffix("\n</tool_call>"))

    let tool1 = try decoder.decode(
        Tool.self,
        from: Data(result1.trimmingToolCallMarkup().utf8)
    )
    let expected1 = Tool(
        name: "get_current_weather",
        arguments: [
            "location": "Paris, FR",
            "unit": "fahrenheit",
        ]
    )

    #expect(tool1 == expected1)

    let result2 = try await llm.request(.init(
        messages: [
            [
                "role": "system",
                "content": "You are weather fetching assistant. Your only purpose is to fetch weather data.",
            ],
            ["role": "system", "content": "The user prefers C°."],
            ["role": "user", "content": "What is weather in Paris like?"],
        ],
        tools: [toolSpec]
    ))

    #expect(result2.hasPrefix("<tool_call>\n"))
    #expect(result2.hasSuffix("\n</tool_call>"))

    let tool2 = try decoder.decode(
        Tool.self,
        from: Data(result2.trimmingToolCallMarkup().utf8)
    )
    let expected2 = Tool(
        name: "get_current_weather",
        arguments: [
            "location": "Paris, FR",
            "unit": "celsius",
        ]
    )

    #expect(tool2 == expected2)
}

private extension String {
    func trimmingToolCallMarkup() -> String {
        let prefix = "<tool_call>\n"
        let suffix = "\n</tool_call>"

        var copy = self
        copy.removeFirst(prefix.count)
        copy.removeLast(suffix.count)
        return copy
    }
}
