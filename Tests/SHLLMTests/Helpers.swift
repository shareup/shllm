import Foundation
import SHLLM

func loadModel<M>(
    directory: @autoclosure () throws -> URL,
    input: @autoclosure () -> UserInput,
    maxOutputTokenCount: @autoclosure () -> Int? = nil
) throws -> LLM<M>? {
    #if targetEnvironment(simulator)
        Swift.print("⚠️ LLMs are not supported in the Simulator")
        return nil
    #else
        do {
            return try LLM(
                directory: directory(),
                input: input(),
                maxOutputTokenCount: maxOutputTokenCount()
            )
        } catch let SHLLMError.directoryNotFound(name) {
            Swift.print("⚠️ \(name) does not exist")
            return nil
        } catch let SHLLMError.missingBundle(name) {
            Swift.print("⚠️ \(name) bundle does not exist")
            return nil
        } catch {
            throw error
        }
    #endif
}

let weatherToolFunction = ToolFunction(
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
)

enum WeatherTool: Codable, CustomStringConvertible, Hashable {
    case getCurrentWeather(GetCurrentWeatherArguments)

    private enum CodingKeys: String, CodingKey {
        case name
        case arguments
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let toolName = try container.decode(String.self, forKey: .name)

        switch toolName {
        case "get_current_weather":
            let args = try container.decode(
                GetCurrentWeatherArguments.self,
                forKey: .arguments
            )
            self = .getCurrentWeather(args)

        default:
            throw DecodingError.dataCorruptedError(
                forKey: CodingKeys.name,
                in: container,
                debugDescription: "Unrecognized tool name: \(toolName)"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case let .getCurrentWeather(args):
            try container.encode("getCurrentWeather", forKey: .name)
            try container.encode(args, forKey: .arguments)
        }
    }

    var description: String {
        switch self {
        case let .getCurrentWeather(args):
            "getCurrentWeather(\(args))"
        }
    }
}

struct GetCurrentWeatherArguments: Codable, CustomStringConvertible, Hashable, Sendable {
    var location: String
    var unit: WeatherUnit

    init(location: String, unit: WeatherUnit) {
        self.location = location
        self.unit = unit
    }

    var description: String {
        "'\(location)', '\(unit)'"
    }
}

enum WeatherUnit: String, Codable, CustomStringConvertible {
    case celsius
    case fahrenheit

    var description: String {
        switch self {
        case .celsius: "Celsius"
        case .fahrenheit: "Fahrenheit"
        }
    }
}
