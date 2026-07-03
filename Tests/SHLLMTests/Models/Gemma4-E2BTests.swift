import Foundation
import MLXVLM
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Gemma4_E2BTests {
    @Test
    func canStreamResult() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try gemma4_E2B(input: input) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(!response.isEmpty)
    }

    @Test
    func canAwaitResult() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try gemma4_E2B(input: input) else { return }

        let response = try await llm.text.result

        Swift.print(response)
        #expect(!response.isEmpty)
    }

    @Test()
    @MainActor
    func canExtractTextFromImageData() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let data = try authenticationFactors
        guard let llm = try gemma4_E2B(image: data) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(response.contains("The 3 authentication factors"))
        #expect(response.contains("Something you forgot"))
        #expect(response.contains("Something you left in the taxi"))
        #expect(response.contains("Something that can be chopped off"))
    }

    @Test()
    @MainActor
    func canExtractTextFromImageURL() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let url = try authenticationFactorsURL
        guard let llm = try gemma4_E2B(image: url) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(response.contains("The 3 authentication factors"))
        #expect(response.contains("Something you forgot"))
        #expect(response.contains("Something you left in the taxi"))
        #expect(response.contains("Something that can be chopped off"))
    }

    @Test
    func canFetchTheWeather() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let input = UserInput(
            chat: [
                .system(
                    "You are a weather assistant who must use the get_current_weather tool to fetch weather data for any location the user asks about.<|think|>"
                ),
                .user("What is the weather in Paris, France?"),
            ],
            additionalContext: ["enable_thinking": true]
        )

        guard let llm = try gemma4_E2B(input, tools: [weatherTool]) else { return }

        var reply = ""
        var toolCallCount = 0
        var weatherLocationFound = false

        for try await response in llm {
            switch response {
            case .reasoning:
                break
            case let .text(text):
                reply.append(text)
            case let .toolCall(toolCall):
                toolCallCount += 1
                #expect(toolCall.function.name == "get_current_weather")

                if case let .string(location) = toolCall.function.arguments["location"] {
                    weatherLocationFound = location.lowercased().contains("paris")
                }
            }
        }

        #expect(reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(toolCallCount == 1)
        #expect(weatherLocationFound)
    }

    @Test
    func canUseStockToolAndRespond() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let chat: [Chat.Message] = [
            .system("""
            <|think|>
            You are a helpful assistant that can provide stock prices.
            When asked for a stock price, you must use the get_stock_price tool.
            <|think|>
            """),
            .user("What is the price of AAPL?"),
        ]

        var input = UserInput(
            chat: chat,
            additionalContext: ["enable_thinking": true]
        )

        guard let llm1 = try gemma4_E2B(input, tools: [stockTool]) else { return }

        var (reasoning, text, toolCallsOpt) = try await llm1.result
        let toolCall = try #require(toolCallsOpt?.first)

        Swift.print("<thinking>\(reasoning ?? "")<thinking>\n\(text ?? "")")
        #expect(reasoning != nil)
        #expect(text == nil)
        #expect(toolCall.function.name == "get_stock_price")
        #expect(toolCall.function.arguments["symbol"] == .string("AAPL"))

        input.appendAssistantToolCall(toolCall)
        input.appendToolResult(["price": 123.45])

        guard let llm2 = try gemma4_E2B(input, tools: [stockTool]) else { return }

        (reasoning, text, toolCallsOpt) = try await llm2.result
        let result = try #require(text)
        Swift.print("<thinking>\(reasoning ?? "")<thinking>\n\(result)")
        #expect(!result.isEmpty)
        #expect(result.lowercased().contains("aapl"))
        #expect(result.contains("123.45"))
    }
}

private extension Gemma4_E2BTests {
    func gemma4_E2B(
        _ input: UserInput,
        tools: [any ToolProtocol] = []
    ) throws -> LLM<Gemma4>? {
        try loadModel(
            directory: LLM.gemma4_E2B,
            input: input,
            tools: tools,
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<turn|>"]
                return config
            },
            responseParser: LLM<Gemma4>.gemma4Parser
        )
    }

    func gemma4_E2B(
        input: UserInput
    ) throws -> LLM<Gemma4>? {
        try gemma4_E2B(input, tools: [])
    }

    func gemma4_E2B(
        image: Data
    ) throws -> LLM<Gemma4>? {
        try loadModel(
            directory: LLM.gemma4_E2B,
            input: imageInput(image),
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<turn|>"]
                return config
            },
            responseParser: LLM<Gemma4>.gemma4Parser
        )
    }

    func gemma4_E2B(
        image: URL
    ) throws -> LLM<Gemma4>? {
        try loadModel(
            directory: LLM.gemma4_E2B,
            input: imageInput(image),
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<turn|>"]
                return config
            },
            responseParser: LLM<Gemma4>.gemma4Parser
        )
    }

    var authenticationFactorsURL: URL {
        get throws {
            guard let url = Bundle.module.url(
                forResource: "3-authentication-factors",
                withExtension: "png"
            ) else {
                throw NSError(
                    domain: NSURLErrorDomain,
                    code: NSURLErrorFileDoesNotExist,
                    userInfo: nil
                )
            }
            return url
        }
    }

    var authenticationFactors: Data {
        get throws {
            try Data(contentsOf: authenticationFactorsURL)
        }
    }
}
