import Foundation
import MLXLMCommon
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
                generateParameters: LLM<M>.generateParameters,
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

extension String {
    func contains(oneOf substrings: [String]) -> Bool {
        for substring in substrings {
            if localizedCaseInsensitiveContains(substring) {
                return true
            }
        }
        return false
    }
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

// MARK: - Multi-step workflow tools (search -> fetch -> find email -> send email)

struct WebSearchInput: Codable, Hashable, Sendable {
    let query: String
}

struct WebSearchOutput: Codable, Hashable, Sendable {
    struct Result: Codable, Hashable, Sendable {
        let title: String
        let url: String
    }

    let results: [Result]
}

let webSearchTool = Tool<WebSearchInput, WebSearchOutput>(
    name: "web_search",
    description: """
    Search the web for relevant pages. 
    Only returns the links to the pages, not the page content itself.
    """,
    parameters: [
        .required("query", type: .string, description: "Search query"),
    ]
) { _ in
    WebSearchOutput(results: [])
}

struct FetchPageInput: Codable, Hashable, Sendable {
    let url: String
}

struct FetchPageOutput: Codable, Hashable, Sendable {
    let content: String
}

let fetchPageTool = Tool<FetchPageInput, FetchPageOutput>(
    name: "fetch_web_page",
    description: "Fetch a web page and return raw text content",
    parameters: [
        .required("url", type: .string, description: "URL to fetch"),
    ]
) { _ in
    FetchPageOutput(content: "")
}

struct FindEmailInput: Codable, Hashable, Sendable {
    let name: String
}

struct FindEmailOutput: Codable, Hashable, Sendable {
    let email: String
}

let findEmailTool = Tool<FindEmailInput, FindEmailOutput>(
    name: "find_email_in_contacts",
    description: "Find a person's email address in the user's contacts list",
    parameters: [
        .required("name", type: .string, description: "Full name to search for"),
    ]
) { _ in
    FindEmailOutput(email: "")
}

struct SendEmailInput: Codable, Hashable, Sendable {
    let to: String
    let subject: String
    let body: String
}

struct SendEmailOutput: Codable, Hashable, Sendable {
    let status: String
}

let sendEmailTool = Tool<SendEmailInput, SendEmailOutput>(
    name: "send_email",
    description: "Send an email to the specified recipient",
    parameters: [
        .required("to", type: .string, description: "Recipient email"),
        .required("subject", type: .string, description: "Email subject"),
        .required("body", type: .string, description: "Email body"),
    ]
) { _ in
    SendEmailOutput(status: "sent")
}
