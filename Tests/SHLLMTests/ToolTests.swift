import Foundation
import MLXLLM
import MLXLMCommon
@testable import SHLLM
import Testing

@Suite(.serialized)
struct ToolTests {
    @Test
    func canCreateTool() throws {
        struct WeatherInput: Codable {
            let location: String
            let unit: String?
        }

        struct WeatherOutput: Codable {
            let temperature: Double
            let conditions: String
        }

        let tool = Tool<WeatherInput, WeatherOutput>(
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
            WeatherOutput(temperature: 22.0, conditions: "Sunny")
        }

        #expect(tool.name == "get_current_weather")
        #expect(tool.type == "function")
        #expect(tool.description == "Get the current weather in a given location")

        let parameters = try #require(tool.parameters)
        #expect(parameters.count == 2)

        let locationParam = parameters.first { $0.name == "location" }
        let unitParam = parameters.first { $0.name == "unit" }

        #expect(locationParam != nil)
        #expect(locationParam?.isRequired == true)
        #expect(locationParam?.type == .string)
        #expect(locationParam?.description == "The city and state, e.g. San Francisco, CA")

        #expect(unitParam != nil)
        #expect(unitParam?.isRequired == false)
        #expect(unitParam?.type == .string)
        #expect(unitParam?.description == "Temperature unit")
    }

    @Test
    func canUseNewResponseAPI() async throws {
        let input = UserInput(chat: [
            .system(
                "You are a weather assistant who can fetch current weather data using the get_current_weather tool."
            ),
            .user("What is the weather in San Francisco?"),
        ])

        guard let llm: LLM<Qwen3Model> = try loadModel(
            directory: LLM<Qwen3Model>.qwen3__1_7B,
            input: input,
            tools: [weatherTool]
        ) else { return }

        var reply = ""
        var toolCallCount = 0

        for try await response in llm {
            switch response {
            case let .text(text):
                reply.append(text)
                #expect(!reply.isEmpty)
            case let .toolCall(toolCall):
                toolCallCount += 1
                #expect(toolCall.function.name == "get_current_weather")
            case .reasoning:
                break
            }
        }

        #expect(!reply.isEmpty) // Qwen 3 thinks before answering
        #expect(toolCallCount == 1)
    }

    @Test
    func jsonIntegration() throws {
        struct ComplexInput: Codable {
            let query: String
            let filters: [String]
            let pagination: PaginationInput
        }

        struct PaginationInput: Codable {
            let page: Int
            let limit: Int
        }

        struct ComplexOutput: Codable {
            let results: [String]
            let totalCount: Int
        }

        let tool = Tool<ComplexInput, ComplexOutput>(
            name: "complex_search",
            description: "Perform a complex search with filters and pagination",
            parameters: [
                .required("query", type: .string, description: "Search query"),
                .required(
                    "filters",
                    type: .array(elementType: .string),
                    description: "Filter criteria"
                ),
                .required(
                    "pagination",
                    type: .object(properties: [
                        .required("page", type: .int, description: "Page number"),
                        .required("limit", type: .int, description: "Items per page"),
                    ]),
                    description: "Pagination settings"
                ),
            ]
        ) { _ in
            ComplexOutput(results: ["result1", "result2"], totalCount: 42)
        }

        #expect(tool.type == "function")
        #expect(tool.name == "complex_search")
        #expect(tool.description == "Perform a complex search with filters and pagination")

        let schema = tool.schema as NSDictionary
        let function = try #require(schema["function"] as? [String: Any])
        let parameters = try #require(function["parameters"] as? [String: Any])
        let properties = try #require(parameters["properties"] as? [String: Any])

        let queryProp = try #require(properties["query"] as? [String: Any])
        #expect(queryProp["type"] as? String == "string")
        #expect(queryProp["description"] as? String == "Search query")

        let filtersProp = try #require(properties["filters"] as? [String: Any])
        #expect(filtersProp["type"] as? String == "array")
        let filtersItems = try #require(filtersProp["items"] as? [String: Any])
        #expect(filtersItems["type"] as? String == "string")

        let paginationProp = try #require(properties["pagination"] as? [String: Any])
        #expect(paginationProp["type"] as? String == "object")
        let paginationProps = try #require(paginationProp["properties"] as? [String: Any])

        let pageProp = try #require(paginationProps["page"] as? [String: Any])
        #expect(pageProp["type"] as? String == "integer")

        let limitProp = try #require(paginationProps["limit"] as? [String: Any])
        #expect(limitProp["type"] as? String == "integer")

        let required = try #require(parameters["required"] as? [String])
        #expect(required.contains("query"))
        #expect(required.contains("filters"))
        #expect(required.contains("pagination"))
    }
}
