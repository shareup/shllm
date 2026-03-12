import Foundation
import MLXLMCommon
import MLXVLM
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Ministral_3_14BTests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try ministral(input) else { return }

        var reasoning = ""
        var result = ""
        for try await reply in llm {
            switch reply {
            case let .reasoning(text):
                reasoning.append(text)
            case let .text(text):
                result.append(text)
            case .toolCall:
                Issue.record()
            }
        }

        if !reasoning.isEmpty {
            Swift.print("[THINK]\n\(reasoning)\n[/THINK]")
        }

        Swift.print(result)
        #expect(!result.isEmpty)
    }

    @Test
    func canStreamTextResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try ministral(input) else { return }

        var result = ""
        for try await reply in llm.text {
            result.append(reply)
        }

        Swift.print(result)
        #expect(!result.isEmpty)
    }

    @Test
    func canAwaitResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try ministral(input) else { return }

        let (_, _text, toolCalls) = try await llm.result

        let text = try #require(_text)
        Swift.print(text)
        #expect(!text.isEmpty)

        #expect(toolCalls == nil)
    }

    @Test
    func canAwaitTextResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try ministral(input) else { return }

        let result = try await llm.text.result
        Swift.print(result)
        #expect(!result.isEmpty)
    }

    @Test
    func canFetchTheWeather() async throws {
        let input = UserInput(chat: [
            .system(
                "You are a weather assistant who must use the get_current_weather tool to fetch weather data for any location the user asks about."
            ),
            .user("What is the weather in Paris, France?"),
        ])

        guard let llm = try ministral(
            input,
            tools: [weatherTool]
        ) else { return }

        var reasoning = ""
        var reply = ""
        var toolCallCount = 0
        var weatherLocationFound = false

        for try await response in llm {
            switch response {
            case let .reasoning(text):
                reasoning.append(text)
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
        #expect(toolCallCount >= 1)
        #expect(weatherLocationFound)
    }

    @Test
    func canChooseBetweenDifferentTools() async throws {
        let input = UserInput(chat: [
            .system(
                "You are a helpful assistant that can provide weather, stock prices, and news."
            ),
            .user("Get the latest news about Apple, sorted by popularity."),
        ])

        guard let llm = try ministral(
            input,
            tools: [weatherTool, stockTool, newsTool]
        ) else { return }

        var reasoning = ""
        var reply = ""
        var toolCallCount = 0
        var newsQueryFound = false
        var newsSortByFound = false

        for try await response in llm {
            switch response {
            case let .reasoning(text):
                reasoning.append(text)
            case let .text(text):
                reply.append(text)
            case let .toolCall(toolCall):
                toolCallCount += 1
                #expect(toolCall.function.name == "get_latest_news")

                if case let .string(query) = toolCall.function.arguments["query"] {
                    newsQueryFound = query.lowercased().contains("apple")
                }
                if case let .string(sortBy) = toolCall.function.arguments["sortBy"] {
                    newsSortByFound = sortBy.lowercased().contains("popularity")
                }
            }
        }

        #expect(reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(toolCallCount >= 1)
        #expect(newsQueryFound)
        #expect(newsSortByFound)
    }

    @Test
    func canUseStockToolAndRespond() async throws {
        let chat: [Chat.Message] = [
            .system(
                "You are a helpful assistant that can provide stock prices. When asked for a stock price, you must use the get_stock_price tool."
            ),
            .user("What is the price of AAPL?"),
        ]

        var input = UserInput(chat: chat)

        guard let llm1 = try ministral(
            input,
            tools: [stockTool]
        ) else { return }

        let (_, text, toolCallsOpt) = try await llm1.result
        let toolCall = try #require(toolCallsOpt?.first)

        #expect(text == nil)
        #expect(toolCall.function.name == "get_stock_price")
        #expect(toolCall.function.arguments["symbol"] == .string("AAPL"))

        input.appendToolResult(["price": 123.45])

        guard let llm2 = try ministral(
            input,
            tools: [stockTool]
        ) else { return }

        let result = try await llm2.text.result
        Swift.print(result)
        #expect(!result.isEmpty)
        #expect(result.lowercased().contains("aapl"))
        #expect(result.contains("123.45"))
    }

    @Test
    func canCompleteMultiToolWorkflowAndEmail() async throws {
        let chat: [Chat.Message] = [
            .system("""
                You are a helpful assistant that must complete tasks by calling tools \
                in sequence. When asked to find information on the web and email it, \
                you must:

                1) use web_search to find a relevant page
                2) use fetch_web_page to retrieve the page content
                3) use find_email_in_contacts to get the recipient's email
                4) use send_email to send the email with the requested information.
                """
            ),
            .user(
                "Find the keynote date from the ACME Conference website and email it to Alex Example."
            ),
        ]

        var input = UserInput(chat: chat)
        guard let llm = try ministral(input, tools: [
            webSearchTool, fetchPageTool, findEmailTool, sendEmailTool,
        ]) else { return }

        let (_, _, toolCallsOutput1) = try await llm.result
        let toolCall1 = try #require(toolCallsOutput1?.first)
        #expect(toolCall1.function.name == "web_search")

        input.appendToolResult([
            "results": [[
                "title": "ACME Conference 2025 Keynote",
                "url": "https://acme.test/conf",
            ]],
        ])

        guard let llm2 = try ministral(input, tools: [
            webSearchTool, fetchPageTool, findEmailTool, sendEmailTool,
        ]) else { return }
        let (_, _, toolCallsOutput2) = try await llm2.result
        let toolCall2 = try #require(toolCallsOutput2?.first)
        #expect(toolCall2.function.name == "fetch_web_page")

        input.appendToolResult([
            "content": "Welcome to ACME Conf! Keynote date: November 5, 2025.",
        ])

        guard let llm3 = try ministral(input, tools: [
            webSearchTool, fetchPageTool, findEmailTool, sendEmailTool,
        ]) else { return }
        let (_, _, toolCallsOutput3) = try await llm3.result
        #expect(toolCallsOutput3?.count == 1)
        let toolCall3 = try #require(toolCallsOutput3?.first)
        #expect(toolCall3.function.name == "find_email_in_contacts")

        input.appendToolResult([
            "email": "alex@example.com",
        ])

        guard let llm4 = try ministral(input, tools: [
            webSearchTool, fetchPageTool, findEmailTool, sendEmailTool,
        ]) else { return }
        let (reasoning, text, toolCalls4) = try await llm4.result

        guard let toolCall4 = toolCalls4?.first else {
            Issue.record("""
                Did not call send_email: reasoning=\(String(describing: reasoning)), \
                text=\(String(describing: text))
                """
            )
            return
        }

        #expect(toolCall4.function.name == "send_email")
        let toArg = try #require(toolCall4.function.arguments["to"])
        let subjectArg = try #require(toolCall4.function.arguments["subject"])
        let bodyArg = try #require(toolCall4.function.arguments["body"])
        #expect((toArg.anyValue as? String) == "alex@example.com")
        #expect((subjectArg.anyValue as? String)?.isEmpty == false)
        #expect((bodyArg.anyValue as? String)?.isEmpty == false)

        input.appendToolResult(["status": "sent"])

        guard let llm5 = try ministral(input, tools: [
            webSearchTool, fetchPageTool, findEmailTool, sendEmailTool,
        ]) else { return }

        let response = try await llm5.text.result
        Swift.print(response)
        #expect(!response.isEmpty)
        #expect(response.contains(oneOf: ["sent", "emailed"]))
        #expect(response.lowercased().contains("alex"))
    }

    @Test
    @MainActor
    func canExtractTextFromImageData() async throws {
        let data = try authenticationFactors
        guard let llm = try ministral(image: data) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(response.contains("authentication"))
    }

    @Test
    @MainActor
    func canExtractTextFromImageURL() async throws {
        let url = try authenticationFactorsURL
        guard let llm = try ministral(image: url) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(response.contains("authentication"))
    }
}

private func ministral(
    _ input: UserInput,
    tools: [any ToolProtocol] = []
) throws -> LLM<Mistral3VLM>? {
    try loadModel(
        directory: LLM<Mistral3VLM>.ministral_3_14B,
        input: input,
        tools: tools,
        responseParser: LLM<Mistral3VLM>.mistral3Parser
    )
}

private func ministral(
    image: Data
) throws -> LLM<Mistral3VLM>? {
    try loadModel(
        directory: LLM<Mistral3VLM>.ministral_3_14B,
        input: imageInput(image),
        responseParser: LLM<Mistral3VLM>.mistral3Parser
    )
}

private func ministral(
    image: URL
) throws -> LLM<Mistral3VLM>? {
    try loadModel(
        directory: LLM<Mistral3VLM>.ministral_3_14B,
        input: imageInput(image),
        responseParser: LLM<Mistral3VLM>.mistral3Parser
    )
}

private var authenticationFactorsURL: URL {
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

private var authenticationFactors: Data {
    get throws {
        try Data(contentsOf: authenticationFactorsURL)
    }
}
