@testable import SHLLM
import Foundation
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

    enum Tool: Codable, Hashable {
        case getCurrentWeather(location: String, unit: WeatherUnit)

        private enum CodingKeys: String, CodingKey {
            case name
            case arguments
        }

        // Decoding: Look at the "name" field and then decode arguments accordingly.
        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let toolName = try container.decode(String.self, forKey: .name)
            let arguments = try container.decode([String: String].self, forKey: .arguments)

            switch toolName {
            case "get_current_weather":
                guard let location = arguments["location"] else {
                    throw DecodingError.keyNotFound(
                        CodingKeys.arguments,
                        DecodingError.Context(codingPath: [CodingKeys.arguments],
                                              debugDescription: "Missing 'location' key")
                    )
                }
                guard let unitString = arguments["unit"],
                      let unit = WeatherUnit(rawValue: unitString) else {
                    throw DecodingError.dataCorruptedError(
                        forKey: CodingKeys.arguments,
                        in: container,
                        debugDescription: "Missing or invalid 'unit' key"
                    )
                }
                self = .getCurrentWeather(location: location, unit: unit)

            default:
                throw DecodingError.dataCorruptedError(
                    forKey: CodingKeys.name,
                    in: container,
                    debugDescription: "Unrecognized tool name: \(toolName)"
                )
            }
        }

        // Encoding: Create a dictionary with "name" and "arguments" based on the enum case.
        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case .getCurrentWeather(let location, let unit):
                try container.encode("getCurrentWeather", forKey: .name)
                let arguments: [String: String] = [
                    "location": location,
                    "unit": unit.rawValue
                ]
                try container.encode(arguments, forKey: .arguments)
            }
        }
    }

    enum WeatherUnit: String, Codable {
        case celsius
        case fahrenheit
    }

    let tools = Tools([
        .init(
            name: "get_current_weather",
            description: "Get the current weather in a given location",
            parameters: [
                .string(name: "location", description: "The city and state, e.g. San Francisco, CA", required: true),
                .string(name: "unit", restrictTo: ["celsius", "fahrenheit"])
            ]
        )
    ])

    let tool1: Tool = try await llm.request(
        tools: tools,
        messages: [
            [
                "role": "system",
                "content": "You are a weather fetching assistant. Your only purpose is to fetch weather data.",
            ],
            ["role": "user", "content": "What is weather in Paris like?"],
        ]
    )

    let expectedTool1 = Tool.getCurrentWeather(location: "Paris, France", unit: .fahrenheit)

    #expect(tool1 == expectedTool1)

    let tool2: Tool = try await llm.request(
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

    let expectedTool2 = Tool.getCurrentWeather(location: "Paris, France", unit: .celsius)

    #expect(tool2 == expectedTool2)
}
