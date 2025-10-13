import Foundation
import protocol MLXLMCommon.LanguageModel
import SHLLM

func loadModel<M: LanguageModel>(
    directory: @autoclosure () throws -> URL,
    input: @autoclosure () -> UserInput,
    processing: @autoclosure (() -> UserInput.Processing?) = nil,
    tools: [any ToolProtocol] = [],
    maxOutputTokenCount: @autoclosure () -> Int? = nil,
    customConfiguration: LLM.CustomConfiguration? = nil,
    responseParser: LLM<M>.ResponseParser = LLM<M>.defaultParser
) throws -> LLM<M>? {
    #if targetEnvironment(simulator)
        Swift.print("⚠️ LLMs are not supported in the Simulator")
        return nil
    #else
        do {
            return LLM(
                directory: try directory(),
                input: input(),
                processing: processing(),
                tools: tools,
                maxOutputTokenCount: maxOutputTokenCount(),
                customConfiguration: customConfiguration,
                responseParser: responseParser
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

func imageInput(
    _ image: Data,
    message: String = "Extract the text in this image."
) -> UserInput {
    UserInput(chat: [
        .system(
            "You are an image understanding model capable of describing the salient features of any image."
        ),
        .user(message, images: [.ciImage(.init(data: image)!)]),
    ])
}

func imageInput(
    _ image: URL,
    message: String = "Extract the text in this image."
) -> UserInput {
    UserInput(chat: [
        .system(
            "You are an image understanding model capable of describing the salient features of any image."
        ),
        .user(message, images: [.url(image)]),
    ])
}

struct WeatherArguments: Codable, CustomStringConvertible, Hashable, Sendable {
    var location: String
    var unit: String?

    init(location: String, unit: String? = nil) {
        self.location = location
        self.unit = unit
    }

    var description: String {
        "'\(location)', unit: \(unit ?? "nil")"
    }
}

struct WeatherResponse: Codable, Hashable, Sendable {
    let temperature: Double
    let conditions: String
}

let weatherTool = Tool<WeatherArguments, WeatherResponse>(
    name: "get_current_weather",
    description: "Get the current weather in a given location",
    parameters: [
        .required(
            "location",
            type: .string,
            description: "The city and state, e.g. San Francisco, CA"
        ),
        .optional(
            "unit",
            type: .string,
            description: "Temperature unit",
            extraProperties: ["enum": ["celsius", "fahrenheit"]]
        ),
    ]
) { _ in
    WeatherResponse(temperature: 22.0, conditions: "Sunny")
}
