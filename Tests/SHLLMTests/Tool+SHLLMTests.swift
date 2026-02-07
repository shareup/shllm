import Foundation
import MLXLMCommon
@testable import SHLLM
import Testing

@Suite(.serialized)
struct ToolSHLLMTests {
    @Test
    func toolProtocolTypeProperty() throws {
        let tool = Tool<EmptyInput, EmptyOutput>(
            name: "test_tool",
            description: "A test tool",
            parameters: []
        ) { _ in EmptyOutput() }

        #expect(tool.type == "function")
    }

    @Test
    func toolProtocolNameProperty() throws {
        let tool = Tool<EmptyInput, EmptyOutput>(
            name: "get_weather",
            description: "Gets weather data",
            parameters: []
        ) { _ in EmptyOutput() }

        #expect(tool.name == "get_weather")
    }

    @Test
    func toolProtocolDescriptionProperty() throws {
        let tool = Tool<EmptyInput, EmptyOutput>(
            name: "test_tool",
            description: "This is a test tool for validation",
            parameters: []
        ) { _ in EmptyOutput() }

        #expect(tool.description == "This is a test tool for validation")
    }

    @Test
    func toolProtocolParametersProperty() throws {
        let tool = Tool<TestInput, EmptyOutput>(
            name: "test_tool",
            description: "A test tool",
            parameters: [
                .required("query", type: .string, description: "The search query"),
                .optional("limit", type: .int, description: "Result limit"),
            ]
        ) { _ in EmptyOutput() }

        let parameters = try #require(tool.parameters)
        #expect(parameters.count == 2)

        let queryParam = parameters.first { $0.name == "query" }
        let limitParam = parameters.first { $0.name == "limit" }

        #expect(queryParam != nil)
        #expect(queryParam?.isRequired == true)
        #expect(queryParam?.type == .string)
        #expect(queryParam?.description == "The search query")

        #expect(limitParam != nil)
        #expect(limitParam?.isRequired == false)
        #expect(limitParam?.type == .int)
        #expect(limitParam?.description == "Result limit")
    }

    @Test
    func parametersWithComplexTypes() throws {
        let tool = Tool<ComplexInput, EmptyOutput>(
            name: "complex_tool",
            description: "A tool with complex parameters",
            parameters: [
                .required("tags", type: .array(elementType: .string), description: "Tag array"),
                .required("settings", type: .object(properties: [
                    .required("enabled", type: .bool, description: "Whether enabled"),
                    .optional("threshold", type: .double, description: "Threshold value"),
                ]), description: "Settings object"),
                .optional("data", type: .data, description: "Binary data"),
            ]
        ) { _ in EmptyOutput() }

        let parameters = try #require(tool.parameters)
        #expect(parameters.count == 3)

        let tagsParam = parameters.first { $0.name == "tags" }
        #expect(tagsParam?.type == .array(elementType: .string))

        let settingsParam = parameters.first { $0.name == "settings" }
        if case let .object(properties) = settingsParam?.type {
            #expect(properties.count == 2)
            let enabledProp = properties.first { $0.name == "enabled" }
            let thresholdProp = properties.first { $0.name == "threshold" }
            #expect(enabledProp?.type == .bool)
            #expect(enabledProp?.isRequired == true)
            #expect(thresholdProp?.type == .double)
            #expect(thresholdProp?.isRequired == false)
        } else {
            Issue.record("Expected object type for settings parameter")
        }

        let dataParam = parameters.first { $0.name == "data" }
        #expect(dataParam?.type == .data)
        #expect(dataParam?.isRequired == false)
    }

    @Test
    func parametersWithExtraProperties() throws {
        let tool = Tool<TestInput, EmptyOutput>(
            name: "test_tool",
            description: "A test tool",
            parameters: [
                .required(
                    "status",
                    type: .string,
                    description: "Status value",

                    extraProperties: ["enum": ["active", "inactive", "pending"]]
                ),
                .optional(
                    "score",
                    type: .double,
                    description: "Score value",
                    extraProperties: ["minimum": 0.0, "maximum": 100.0]
                ),
            ]
        ) { _ in EmptyOutput() }

        let parameters = try #require(tool.parameters)
        #expect(parameters.count == 2)

        let statusParam = parameters.first { $0.name == "status" }
        let scoreParam = parameters.first { $0.name == "score" }

        #expect(
            statusParam?
                .extraProperties["enum"] as? [String] == ["active", "inactive", "pending"]
        )
        #expect(scoreParam?.extraProperties["minimum"] as? Double == 0.0)
        #expect(scoreParam?.extraProperties["maximum"] as? Double == 100.0)
    }

    @Test
    func toolParameterTypeEquality() {
        #expect(ToolParameterType.string == ToolParameterType.string)
        #expect(ToolParameterType.bool == ToolParameterType.bool)
        #expect(ToolParameterType.int == ToolParameterType.int)
        #expect(ToolParameterType.double == ToolParameterType.double)
        #expect(ToolParameterType.data == ToolParameterType.data)

        #expect(ToolParameterType.string != ToolParameterType.bool)
        #expect(ToolParameterType.int != ToolParameterType.double)
        #expect(ToolParameterType.string != ToolParameterType.int)
        #expect(ToolParameterType.bool != ToolParameterType.double)
        #expect(ToolParameterType.data != ToolParameterType.string)

        let arrayType1 = ToolParameterType.array(elementType: .string)
        let arrayType2 = ToolParameterType.array(elementType: .string)
        let arrayType3 = ToolParameterType.array(elementType: .int)
        let arrayType4 = ToolParameterType.array(elementType: .bool)

        #expect(arrayType1 == arrayType2)
        #expect(arrayType1 != arrayType3)
        #expect(arrayType3 != arrayType4)

        let nestedArrayType1 = ToolParameterType
            .array(elementType: .array(elementType: .string))
        let nestedArrayType2 = ToolParameterType
            .array(elementType: .array(elementType: .string))
        let nestedArrayType3 = ToolParameterType.array(elementType: .array(elementType: .int))

        #expect(nestedArrayType1 == nestedArrayType2)
        #expect(nestedArrayType1 != nestedArrayType3)

        #expect(arrayType1 != ToolParameterType.string)
        #expect(ToolParameterType.int != arrayType3)

        let objectType1 = ToolParameterType.object(properties: [
            ToolParameter.required("name", type: .string, description: "Name"),
        ])
        let objectType2 = ToolParameterType.object(properties: [
            ToolParameter.required("name", type: .string, description: "Name"),
        ])
        let objectType3 = ToolParameterType.object(properties: [
            ToolParameter.required("id", type: .int, description: "ID"),
        ])

        #expect(objectType1 == objectType2)
        #expect(objectType1 != objectType3)

        let multiPropObject1 = ToolParameterType.object(properties: [
            ToolParameter.required("name", type: .string, description: "Name"),
            ToolParameter.optional("age", type: .int, description: "Age"),
        ])
        let multiPropObject2 = ToolParameterType.object(properties: [
            ToolParameter.required("name", type: .string, description: "Name"),
            ToolParameter.optional("age", type: .int, description: "Age"),
        ])
        let multiPropObject3 = ToolParameterType.object(properties: [
            ToolParameter.required("name", type: .string, description: "Name"),
            ToolParameter.optional("score", type: .double, description: "Score"),
        ])

        #expect(multiPropObject1 == multiPropObject2)
        #expect(multiPropObject1 != multiPropObject3)

        #expect(objectType1 != ToolParameterType.string)
        #expect(objectType1 != arrayType1)

        let emptyObject1 = ToolParameterType.object(properties: [])
        let emptyObject2 = ToolParameterType.object(properties: [])
        #expect(emptyObject1 == emptyObject2)
        #expect(emptyObject1 != objectType1)

        let nestedObject1 = ToolParameterType.object(properties: [
            ToolParameter.required("config", type: .object(properties: [
                ToolParameter.required("host", type: .string, description: "Host"),
            ]), description: "Config"),
        ])
        let nestedObject2 = ToolParameterType.object(properties: [
            ToolParameter.required("config", type: .object(properties: [
                ToolParameter.required("host", type: .string, description: "Host"),
            ]), description: "Config"),
        ])
        let nestedObject3 = ToolParameterType.object(properties: [
            ToolParameter.required("config", type: .object(properties: [
                ToolParameter.required("port", type: .int, description: "Port"),
            ]), description: "Config"),
        ])

        #expect(nestedObject1 == nestedObject2)
        #expect(nestedObject1 != nestedObject3)
    }

    @Test
    func toolParameterEquality() {
        let param1 = ToolParameter.required("test", type: .string, description: "Test param")
        let param2 = ToolParameter.required("test", type: .string, description: "Test param")
        #expect(param1 == param2)

        let param3 = ToolParameter.required("test", type: .int, description: "Test param")
        #expect(param1 != param3)

        let param4 = ToolParameter.optional("test", type: .string, description: "Test param")
        #expect(param1 != param4)

        let param5 = ToolParameter.required(
            "test",
            type: .string,
            description: "Different desc"
        )
        #expect(param1 != param5)

        let param6 = ToolParameter.required("other", type: .string, description: "Test param")
        #expect(param1 != param6)

        let paramWithExtra1 = ToolParameter.required(
            "test",
            type: .string,
            description: "Test",

            extraProperties: ["enum": ["a", "b"]]
        )
        let paramWithExtra2 = ToolParameter.required(
            "test",
            type: .string,
            description: "Test",

            extraProperties: ["enum": ["a", "b"]]
        )
        #expect(paramWithExtra1 == paramWithExtra2)

        let paramWithExtra3 = ToolParameter.required(
            "test",
            type: .string,
            description: "Test",

            extraProperties: ["enum": ["c", "d"]]
        )
        #expect(paramWithExtra1 != paramWithExtra3)

        let paramWithExtra4 = ToolParameter.required(
            "test",
            type: .string,
            description: "Test",

            extraProperties: ["pattern": "^[a-z]+$"]
        )
        #expect(paramWithExtra1 != paramWithExtra4)

        let paramWithEmptyExtra1 = ToolParameter.required(
            "test",
            type: .string,
            description: "Test",

            extraProperties: [:]
        )
        let paramWithEmptyExtra2 = ToolParameter.required(
            "test",
            type: .string,
            description: "Test",

            extraProperties: [:]
        )
        let paramWithNoExtra = ToolParameter.required(
            "test",
            type: .string,
            description: "Test"
        )

        #expect(paramWithEmptyExtra1 == paramWithEmptyExtra2)
        #expect(
            paramWithEmptyExtra1 ==
                paramWithNoExtra
        )

        let complexExtra1 = ToolParameter.required(
            "test",
            type: .string,
            description: "Test",

            extraProperties: [
                "enum": ["a", "b"],
                "validation": [
                    "minLength": 1,
                    "maxLength": 10,
                ],
            ]
        )
        let complexExtra2 = ToolParameter.required(
            "test",
            type: .string,
            description: "Test",

            extraProperties: [
                "enum": ["a", "b"],
                "validation": [
                    "minLength": 1,
                    "maxLength": 10,
                ],
            ]
        )
        let complexExtra3 = ToolParameter.required(
            "test",
            type: .string,
            description: "Test",

            extraProperties: [
                "enum": ["a", "b"],
                "validation": [
                    "minLength": 2,
                    "maxLength": 10,
                ],
            ]
        )

        #expect(complexExtra1 == complexExtra2)
        #expect(complexExtra1 != complexExtra3)

        let intParam = ToolParameter.required("test", type: .int, description: "Test param")
        let doubleParam = ToolParameter.required(
            "test",
            type: .double,
            description: "Test param"
        )
        let boolParam = ToolParameter.required("test", type: .bool, description: "Test param")
        let dataParam = ToolParameter.required("test", type: .data, description: "Test param")

        #expect(intParam != doubleParam)
        #expect(intParam != boolParam)
        #expect(intParam != dataParam)
        #expect(doubleParam != boolParam)
        #expect(doubleParam != dataParam)
        #expect(boolParam != dataParam)

        let arrayParam1 = ToolParameter.required(
            "test",
            type: .array(elementType: .string),
            description: "Test"
        )
        let arrayParam2 = ToolParameter.required(
            "test",
            type: .array(elementType: .string),
            description: "Test"
        )
        let arrayParam3 = ToolParameter.required(
            "test",
            type: .array(elementType: .int),
            description: "Test"
        )

        #expect(arrayParam1 == arrayParam2)
        #expect(arrayParam1 != arrayParam3)

        let objectParam1 = ToolParameter.required("test", type: .object(properties: [
            ToolParameter.required("name", type: .string, description: "Name"),
        ]), description: "Test")
        let objectParam2 = ToolParameter.required("test", type: .object(properties: [
            ToolParameter.required("name", type: .string, description: "Name"),
        ]), description: "Test")
        let objectParam3 = ToolParameter.required("test", type: .object(properties: [
            ToolParameter.required("id", type: .int, description: "ID"),
        ]), description: "Test")

        #expect(objectParam1 == objectParam2)
        #expect(objectParam1 != objectParam3)
        #expect(arrayParam1 != objectParam1)
    }

    @Test
    func invalidSchemaHandling() {
        let invalidTool = TestInvalidTool()

        #expect(invalidTool.type == nil)
        #expect(invalidTool.name == nil)
        #expect(invalidTool.description == nil)
        #expect(invalidTool.parameters == nil)
    }

    @Test
    func schemaWithMissingFunction() {
        let toolWithMissingFunction = TestToolWithMissingFunction()

        #expect(toolWithMissingFunction.type == "function")
        #expect(toolWithMissingFunction.name == nil)
        #expect(toolWithMissingFunction.description == nil)
        #expect(toolWithMissingFunction.parameters == nil)
    }

    @Test
    func schemaWithInvalidParameters() {
        let toolWithInvalidParams = TestToolWithInvalidParameters()

        #expect(toolWithInvalidParams.type == "function")
        #expect(toolWithInvalidParams.name == "test")
        #expect(toolWithInvalidParams.description == "Test")
        #expect(toolWithInvalidParams.parameters == nil)
    }

    @Test
    func schemaWithSendableDictionaries() throws {
        let tool = TestToolWithSendableSchema()

        #expect(tool.type == "function")
        #expect(tool.name == "manual_tool")
        #expect(tool.description == "Manually defined schema")

        let parameters = try #require(tool.parameters)
        #expect(parameters.count == 2)

        let queryParam = parameters.first { $0.name == "query" }
        let limitParam = parameters.first { $0.name == "limit" }

        #expect(queryParam?.isRequired == true)
        #expect(queryParam?.type == .string)
        #expect(queryParam?.description == "Search query")

        #expect(limitParam?.isRequired == false)
        #expect(limitParam?.type == .int)
        #expect(limitParam?.description == "Result limit")
    }

    @Test
    func nestedObjectParsing() throws {
        let tool = Tool<NestedInput, EmptyOutput>(
            name: "nested_tool",
            description: "A tool with nested objects",
            parameters: [
                .required("config", type: .object(properties: [
                    .required("database", type: .object(properties: [
                        .required("host", type: .string, description: "DB host"),
                        .required("port", type: .int, description: "DB port"),
                        .optional("ssl", type: .bool, description: "Use SSL"),
                    ]), description: "Database config"),
                    .required("cache", type: .object(properties: [
                        .required("ttl", type: .int, description: "Time to live"),
                        .required("maxSize", type: .int, description: "Max cache size"),
                    ]), description: "Cache config"),
                ]), description: "Application config"),
            ]
        ) { _ in EmptyOutput() }

        let parameters = try #require(tool.parameters)
        #expect(parameters.count == 1)

        let configParam = parameters.first { $0.name == "config" }
        #expect(configParam != nil)

        if case let .object(configProps) = configParam?.type {
            #expect(configProps.count == 2)

            let dbParam = configProps.first { $0.name == "database" }
            let cacheParam = configProps.first { $0.name == "cache" }

            #expect(dbParam != nil)
            #expect(cacheParam != nil)

            if case let .object(dbProps) = dbParam?.type {
                #expect(dbProps.count == 3)
                let hostProp = dbProps.first { $0.name == "host" }
                let portProp = dbProps.first { $0.name == "port" }
                let sslProp = dbProps.first { $0.name == "ssl" }

                #expect(hostProp?.type == .string)
                #expect(hostProp?.isRequired == true)
                #expect(portProp?.type == .int)
                #expect(portProp?.isRequired == true)
                #expect(sslProp?.type == .bool)
                #expect(sslProp?.isRequired == false)
            } else {
                Issue.record("Expected object type for database parameter")
            }
        } else {
            Issue.record("Expected object type for config parameter")
        }
    }

    @Test
    func arrayOfComplexTypes() throws {
        let tool = Tool<ArrayInput, EmptyOutput>(
            name: "array_tool",
            description: "A tool with array of objects",
            parameters: [
                .required("items", type: .array(elementType: .object(properties: [
                    .required("id", type: .string, description: "Item ID"),
                    .required("value", type: .double, description: "Item value"),
                    .optional("metadata", type: .object(properties: [
                        .optional(
                            "tags",
                            type: .array(elementType: .string),
                            description: "Tags"
                        ),
                    ]), description: "Item metadata"),
                ])), description: "Array of items"),
            ]
        ) { _ in EmptyOutput() }

        let parameters = try #require(tool.parameters)
        #expect(parameters.count == 1)

        let itemsParam = parameters.first { $0.name == "items" }
        #expect(itemsParam != nil)

        if case let .array(elementType) = itemsParam?.type,
           case let .object(itemProps) = elementType
        {
            #expect(itemProps.count == 3)

            let idProp = itemProps.first { $0.name == "id" }
            let valueProp = itemProps.first { $0.name == "value" }
            let metadataProp = itemProps.first { $0.name == "metadata" }

            #expect(idProp?.type == .string)
            #expect(idProp?.isRequired == true)
            #expect(valueProp?.type == .double)
            #expect(valueProp?.isRequired == true)
            #expect(metadataProp?.isRequired == false)

            if case let .object(metadataProps) = metadataProp?.type {
                #expect(metadataProps.count == 1)
                let tagsProp = metadataProps.first { $0.name == "tags" }
                if case let .array(tagsElementType) = tagsProp?.type {
                    #expect(tagsElementType == .string)
                } else {
                    Issue.record("Expected array type for tags")
                }
            } else {
                Issue.record("Expected object type for metadata")
            }
        } else {
            Issue.record("Expected array of objects type for items parameter")
        }
    }
}

private struct EmptyInput: Codable {}
private struct EmptyOutput: Codable {}

private struct TestInput: Codable {
    let query: String
    let limit: Int?
}

private struct ComplexInput: Codable {
    let tags: [String]
    let settings: Settings
    let data: Data?

    struct Settings: Codable {
        let enabled: Bool
        let threshold: Double?
    }
}

private struct NestedInput: Codable {
    let config: Config

    struct Config: Codable {
        let database: Database
        let cache: Cache

        struct Database: Codable {
            let host: String
            let port: Int
            let ssl: Bool?
        }

        struct Cache: Codable {
            let ttl: Int
            let maxSize: Int
        }
    }
}

private struct ArrayInput: Codable {
    let items: [Item]

    struct Item: Codable {
        let id: String
        let value: Double
        let metadata: Metadata?

        struct Metadata: Codable {
            let tags: [String]?
        }
    }
}

private struct TestInvalidTool: ToolProtocol {
    var schema: ToolSpec {
        ["invalid": "schema"]
    }
}

private struct TestToolWithMissingFunction: ToolProtocol {
    var schema: ToolSpec {
        ["type": "function"]
    }
}

private struct TestToolWithInvalidParameters: ToolProtocol {
    var schema: ToolSpec {
        [
            "type": "function",
            "function": [
                "name": "test",
                "description": "Test",
                "parameters": [
                    "type": "invalid_type",
                    "properties": [:] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }
}

private struct TestToolWithSendableSchema: ToolProtocol {
    var schema: ToolSpec {
        [
            "type": "function",
            "function": [
                "name": "manual_tool",
                "description": "Manually defined schema",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "query": [
                            "type": "string",
                            "description": "Search query",
                        ] as [String: any Sendable],
                        "limit": [
                            "type": "integer",
                            "description": "Result limit",
                        ] as [String: any Sendable],
                    ] as [String: any Sendable],
                    "required": ["query"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }
}
