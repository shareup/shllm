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

struct StockArguments: Codable, CustomStringConvertible, Hashable, Sendable {
    var symbol: String

    init(symbol: String) {
        self.symbol = symbol
    }

    var description: String {
        "'\(symbol)'"
    }
}

struct StockResponse: Codable, Hashable, Sendable {
    let price: Double
}

let stockTool = Tool<StockArguments, StockResponse>(
    name: "get_stock_price",
    description: "Get the current price of a stock",
    parameters: [
        .required(
            "symbol",
            type: .string,
            description: "The stock symbol, e.g. AAPL"
        ),
    ]
) { _ in
    StockResponse(price: 150.0)
}

struct NewsArguments: Codable, CustomStringConvertible, Hashable, Sendable {
    var query: String
    var sortBy: String?

    init(query: String, sortBy: String? = nil) {
        self.query = query
        self.sortBy = sortBy
    }

    var description: String {
        "'\(query)', sortBy: \(sortBy ?? "nil")"
    }
}

struct NewsResponse: Codable, Hashable, Sendable {
    let headlines: [String]
}

let newsTool = Tool<NewsArguments, NewsResponse>(
    name: "get_latest_news",
    description: "Get the latest news headlines",
    parameters: [
        .required(
            "query",
            type: .string,
            description: "The search query for news headlines"
        ),
        .optional(
            "sortBy",
            type: .string,
            description: "Sort by 'relevancy' or 'popularity'",
            extraProperties: ["enum": ["relevancy", "popularity"]]
        ),
    ]
) { _ in
    NewsResponse(headlines: [
        "Apple announces new iPhone",
        "Microsoft releases new Surface Pro",
    ])
}
