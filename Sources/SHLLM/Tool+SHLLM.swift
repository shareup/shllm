import Foundation
import MLXLMCommon

public extension ToolProtocol {
    var type: String? { schema["type"] as? String }

    var name: String? {
        guard let function = schema["function"] as? [String: Any],
              let name = function["name"] as? String
        else { return nil }
        return name
    }

    var description: String? {
        guard let function = schema["function"] as? [String: Any],
              let description = function["description"] as? String
        else { return nil }
        return description
    }

    var parameters: [ToolParameter]? {
        guard let function = schema["function"] as? [String: Any],
              let schemaParameters = function["parameters"] as? [String: Any],
              schemaParameters["type"] as? String == "object",
              let properties = schemaParameters["properties"] as? [String: [String: Any]]
        else { return nil }

        let requiredParams = Set(schemaParameters["required"] as? [String] ?? [])

        var resultParameters = [ToolParameter]()
        for (name, paramSchema) in properties {
            guard let toolParam = Self.parseParameter(
                name: name,
                schema: paramSchema,
                isRequired: requiredParams.contains(name)
            ) else { return nil }
            resultParameters.append(toolParam)
        }

        return resultParameters
    }
}

extension ToolParameterType: @retroactive Equatable {
    public static func == (lhs: ToolParameterType, rhs: ToolParameterType) -> Bool {
        switch (lhs, rhs) {
        case (.string, .string),
             (.bool, .bool),
             (.int, .int),
             (.double, .double),
             (.data, .data):
            true

        case let (.array(lhsElement), .array(rhsElement)):
            lhsElement == rhsElement

        case let (.object(lhsProps), .object(rhsProps)):
            lhsProps == rhsProps

        default:
            false
        }
    }
}

extension ToolParameter: @retroactive Equatable {
    public static func == (lhs: ToolParameter, rhs: ToolParameter) -> Bool {
        guard lhs.name == rhs.name,
              lhs.type == rhs.type,
              lhs.description == rhs.description,
              lhs.isRequired == rhs.isRequired
        else { return false }
        let lhsProps = lhs.extraProperties as NSDictionary
        let rhsProps = rhs.extraProperties as NSDictionary
        return lhsProps.isEqual(to: rhsProps as! [AnyHashable: Any])
    }
}

private extension ToolProtocol {
    static func parseParameter(
        name: String,
        schema: [String: Any],
        isRequired: Bool
    ) -> ToolParameter? {
        guard let description = schema["description"] as? String,
              let type = parseType(from: schema)
        else { return nil }

        var extraProperties = schema
        let standardKeys: [String] = [
            "type",
            "description",
            "items",
            "properties",
            "required",
            "contentEncoding",
        ]
        for key in standardKeys {
            extraProperties.removeValue(forKey: key)
        }

        if isRequired {
            return ToolParameter.required(
                name,
                type: type,
                description: description,
                extraProperties: extraProperties
            )
        } else {
            return ToolParameter.optional(
                name,
                type: type,
                description: description,
                extraProperties: extraProperties
            )
        }
    }

    static func parseType(from schema: [String: Any]) -> ToolParameterType? {
        guard let typeString = schema["type"] as? String else {
            return nil
        }

        switch typeString {
        case "string":
            if schema["contentEncoding"] as? String == "base64" {
                return .data
            }
            return .string

        case "boolean":
            return .bool

        case "integer":
            return .int

        case "number":
            return .double

        case "array":
            guard let itemsSchema = schema["items"] as? [String: Any],
                  let elementType = parseType(from: itemsSchema)
            else {
                return nil
            }
            return .array(elementType: elementType)

        case "object":
            typealias Props = [String: [String: Any]]
            guard let properties = schema["properties"] as? Props else {
                return nil
            }
            let requiredNames = schema["required"] as? [String] ?? []
            let requiredSet = Set(requiredNames)
            var subParameters = [ToolParameter]()

            for (name, propSchema) in properties {
                guard let subParam = parseParameter(
                    name: name,
                    schema: propSchema,
                    isRequired: requiredSet.contains(name)
                ) else { return nil }
                subParameters.append(subParam)
            }
            return .object(properties: subParameters)

        default:
            return nil
        }
    }
}
