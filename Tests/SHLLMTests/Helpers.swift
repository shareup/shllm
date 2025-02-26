import Foundation
import SHLLM

func loadModel<M: InitializableWithDirectory>(
    from directory: @autoclosure () throws -> URL
) async throws -> M? {
    #if targetEnvironment(simulator)
        Swift.print("⚠️ LLMs are not supported in the Simulator")
        return nil
    #else
        do {
            return try await M(directory: directory())
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

protocol InitializableWithDirectory {
    init(directory: URL) async throws
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

enum WeatherTool: Codable, Hashable {
    case getCurrentWeather(location: String, unit: WeatherUnit)

    private enum CodingKeys: String, CodingKey {
        case name
        case arguments
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let toolName = try container.decode(String.self, forKey: .name)
        let arguments = try container.decode([String: String].self, forKey: .arguments)

        switch toolName {
        case "get_current_weather":
            guard let location = arguments["location"] else {
                throw DecodingError.keyNotFound(
                    CodingKeys.arguments,
                    DecodingError.Context(
                        codingPath: [CodingKeys.arguments],
                        debugDescription: "Missing 'location' key"
                    )
                )
            }
            guard let unitString = arguments["unit"],
                  let unit = WeatherUnit(rawValue: unitString)
            else {
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

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case let .getCurrentWeather(location, unit):
            try container.encode("getCurrentWeather", forKey: .name)
            let arguments: [String: String] = [
                "location": location,
                "unit": unit.rawValue,
            ]
            try container.encode(arguments, forKey: .arguments)
        }
    }
}

enum WeatherUnit: String, Codable {
    case celsius
    case fahrenheit
}
