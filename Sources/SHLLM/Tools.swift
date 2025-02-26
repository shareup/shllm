import MLXNN

public struct Tools {
    public let functions: [ToolFunction]

    public init(_ functions: [ToolFunction] = []) {
        self.functions = functions
    }

    public func toSpec() -> [[String: any Sendable]] {
        return functions.map { $0.toSpec() }
    }
}

public struct ToolFunction {
    public let name: String
    public let description: String?
    public let parameters: [ToolFunctionParameter]

    init(name: String, description: String? = nil, parameters: [ToolFunctionParameter]) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    public func toSpec() -> [String: any Sendable] {
        var propertiesSpec = ToolFunctionParameterType.object(properties: parameters).toSpec()

        let required: [String] = parameters.compactMap { $0.required ? $0.name : nil }

        if !required.isEmpty {
            propertiesSpec["required"] = required
        }

        var functionSpec: [String: any Sendable] = [
            "name": name,
            "parameters": propertiesSpec
        ]

        if let description = description {
            functionSpec["description"] = description
        }

        return [
            "type": "function",
            "function": functionSpec
        ]
    }
}

public struct ToolFunctionParameter {
    public let name: String
    public let type: ToolFunctionParameterType
    public let description: String?
    public let required: Bool

    static func string(name: String, description: String? = nil, required: Bool = false, format: ToolFunctionStringFormat = .plain, minLength: Int? = nil, maxLength: Int? = nil, restrictTo: [String]? = nil, asConst: String? = nil) -> ToolFunctionParameter {
        .init(name: name, type: .string(format: format, minLength: minLength, maxLength: maxLength, restrictTo: restrictTo, asConst: asConst), description: description, required: required)
    }

    static func number(name: String, description: String? = nil, required: Bool = false, minimum: Double? = nil, maximum: Double? = nil, asConst: Double? = nil, multipleOf: Double? = nil) -> ToolFunctionParameter {
        .init(name: name, type: .number(minimum: minimum, maximum: maximum, asConst: asConst, multipleOf: multipleOf), description: description, required: required)
    }

    static func integer(name: String, description: String? = nil, required: Bool = false, minimum: Int? = nil, maximum: Int? = nil, asConst: Int? = nil, multipleOf: Int? = nil) -> ToolFunctionParameter {
        .init(name: name, type: .integer(minimum: minimum, maximum: maximum, asConst: asConst, multipleOf: multipleOf), description: description, required: required)
    }

    static func array(name: String, description: String? = nil, required: Bool = false, items: ToolFunctionParameterType? = nil) -> ToolFunctionParameter {
        .init(name: name, type: .array(items: items), description: description, required: required)
    }

    static func object(name: String, description: String? = nil, required: Bool = false, properties: [ToolFunctionParameter] = []) -> ToolFunctionParameter {
        .init(name: name, type: .object(properties: properties), description: description, required: required)
    }

    static func boolean(name: String, description: String? = nil, required: Bool = false) -> ToolFunctionParameter {
        .init(name: name, type: .boolean, description: description, required: required)
    }

    static func null(name: String, description: String? = nil, required: Bool = false) -> ToolFunctionParameter {
        .init(name: name, type: .null, description: description, required: required)
    }

    init(name: String, type: ToolFunctionParameterType, description: String? = nil, required: Bool = false) {
        self.name = name
        self.type = type
        self.description = description
        self.required = required
    }

    public func toSpec() -> [String: any Sendable] {
        var typeSpec = type.toSpec()

        var dict: [String: any Sendable] = [
            "name": name,
            "type": typeSpec["type"]
        ]

        if let description = description {
            dict["description"] = description
        }

        typeSpec.removeValue(forKey: "type")
        typeSpec.forEach { dict[$0.key] = $0.value }

        return dict
    }
}

public indirect enum ToolFunctionParameterType {
    case array(items: ToolFunctionParameterType? = nil)
    case object(properties: [ToolFunctionParameter] = [])
    case number(minimum: Double? = nil, maximum: Double? = nil, asConst: Double? = nil, multipleOf: Double? = nil)
    case integer(minimum: Int? = nil, maximum: Int? = nil, asConst: Int? = nil, multipleOf: Int? = nil)
    case string(format: ToolFunctionStringFormat = .plain, minLength: Int? = nil, maxLength: Int? = nil, restrictTo: [String]? = nil, asConst: String? = nil)
    case boolean
    case null

    public func toSpec() -> [String: any Sendable] {
        switch self {
        case .array(let items):
            var dict: [String: Any] = ["type": "array"]
            if let items = items {
                dict["items"] = items.toSpec()
            }
            return dict

        case .object(let properties):
            var _properties: [String: any Sendable] = [:]

            properties.forEach { prop in
                var dict = prop.toSpec()
                let name = prop.name
                dict.removeValue(forKey: "name")
                _properties[name] = dict
            }

            return [
                "type": "object",
                "properties": _properties,
            ]

        case .number(let minimum, let maximum, let asConst, let multipleOf):
            var dict: [String: Any] = ["type": "number"]
            if let minimum = minimum { dict["minimum"] = minimum }
            if let maximum = maximum { dict["maximum"] = maximum }
            if let asConst = asConst { dict["const"] = asConst }
            if let multipleOf = multipleOf { dict["multipleOf"] = multipleOf }
            return dict

        case .integer(let minimum, let maximum, let asConst, let multipleOf):
            var dict: [String: Any] = ["type": "integer"]
            if let minimum = minimum { dict["minimum"] = minimum }
            if let maximum = maximum { dict["maximum"] = maximum }
            if let asConst = asConst { dict["const"] = asConst }
            if let multipleOf = multipleOf { dict["multipleOf"] = multipleOf }
            return dict

        case .string(let format, let minLength, let maxLength, let restrictTo, let asConst):
            var dict: [String: Any] = ["type": "string"]

            if format != .plain {
                dict["format"] = format.rawValue
            }
            if let minLength = minLength { dict["minLength"] = minLength }
            if let maxLength = maxLength { dict["maxLength"] = maxLength }
            if let restrictTo = restrictTo { dict["enum"] = restrictTo }
            if let asConst = asConst { dict["const"] = asConst }
            return dict

        case .boolean:
            return ["type": "boolean"]

        case .null:
            return ["type": "null"]
        }
    }
}

public enum ToolFunctionStringFormat: String {
    case plain = "plain"
    case date = "date"
}
