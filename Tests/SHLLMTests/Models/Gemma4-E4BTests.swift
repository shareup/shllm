import Foundation
import MLXVLM
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Gemma4_E4BTests {
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

        guard let llm = try gemma4_E4B(input: input) else { return }

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

        guard let llm = try gemma4_E4B(input: input) else { return }

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
        guard let llm = try gemma4_E4B(image: data) else { return }

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
        guard let llm = try gemma4_E4B(image: url) else { return }

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

        guard let llm = try gemma4_E4B(input, tools: [weatherTool]) else { return }

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

        guard let llm1 = try gemma4_E4B(input, tools: [stockTool]) else { return }

        let (reasoning1, text1, toolCallsOpt1) = try await llm1.result
        let toolCall1 = try #require(toolCallsOpt1?.first)

        Swift.print("<thinking>\(reasoning1 ?? "")</thinking>\n\(text1 ?? "")")
        #expect(reasoning1 != nil)
        #expect(text1 == nil)
        #expect(toolCall1.function.name == "get_stock_price")
        #expect(toolCall1.function.arguments["symbol"] == .string("AAPL"))

        input.appendAssistantToolCall(toolCall1)
        input.appendToolResult(["price": 123.45])

        guard let llm2 = try gemma4_E4B(input, tools: [stockTool]) else { return }

        let (reasoning2, text2, toolCallsOpt2) = try await llm2.result
        let result = try #require(text2)
        Swift.print("<thinking>\(reasoning2 ?? "")</thinking>\n\(result)")
        #expect(!result.isEmpty)
        #expect(result.lowercased().contains("aapl"))
        #expect(result.contains("123.45"))
        #expect(toolCallsOpt2 == nil)
    }

    @Test
    func canUseToolsWithNonStringArgumentsAndRespond() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let chat: [Chat.Message] = [
            .system("""
            <|think|>
            You are an email assistant. When asked to read an email, call mail_read exactly once.
            After the tool result is provided, reply with the email subject exactly and do not call tools again.
            <|think|>
            """),
            .user("Read email 158348 from account me@example.com in mailbox INBOX."),
        ]

        var input = UserInput(
            chat: chat,
            additionalContext: ["enable_thinking": true]
        )

        guard let llm1 = try gemma4_E4B(input, tools: [mailReadTool]) else { return }

        let (reasoning1, text1, toolCallsOpt1) = try await llm1.result
        let toolCall1 = try #require(toolCallsOpt1?.first)

        Swift.print("<thinking>\(reasoning1 ?? "")<thinking>\n\(text1 ?? "")")
        #expect(reasoning1 != nil)
        #expect(text1 == nil)
        #expect(toolCall1.function.name == "mail_read")
        #expect(toolCall1.function.arguments["account"] == .string("me@example.com"))
        #expect(toolCall1.function.arguments["mailbox"] == .string("INBOX"))

        let idArgument = try #require(toolCall1.function.arguments["id"])
        #expect(idArgument == .int(158_348))
        #expect(idArgument != .string("158348"))

        input.appendAssistantToolCall(toolCall1)
        input.appendToolResult(MailReadResponse(subject: mailReadSubject))

        guard let llm2 = try gemma4_E4B(input, tools: [mailReadTool]) else { return }

        let (reasoning2, text2, toolCallsOpt2) = try await llm2.result
        let result = try #require(text2)
        Swift.print("<thinking>\(reasoning2 ?? "")<thinking>\n\(result)")
        #expect(result.contains(mailReadSubject))
        #expect(toolCallsOpt2 == nil)
    }
}

private extension Gemma4_E4BTests {
    func gemma4_E4B(
        _ input: UserInput,
        tools: [any ToolProtocol] = []
    ) throws -> LLM<Gemma4>? {
        try loadModel(
            directory: LLM.gemma4_E4B,
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

    func gemma4_E4B(
        input: UserInput
    ) throws -> LLM<Gemma4>? {
        try gemma4_E4B(input, tools: [])
    }

    func gemma4_E4B(
        image: Data
    ) throws -> LLM<Gemma4>? {
        try loadModel(
            directory: LLM.gemma4_E4B,
            input: imageInput(image),
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<turn|>"]
                return config
            },
            responseParser: LLM<Gemma4>.gemma4Parser
        )
    }

    func gemma4_E4B(
        image: URL
    ) throws -> LLM<Gemma4>? {
        try loadModel(
            directory: LLM.gemma4_E4B,
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
