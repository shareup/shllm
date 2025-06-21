import Combine
import Foundation
import MLXLMCommon

public typealias Message = MLXLMCommon.Message
public typealias UserInput = MLXLMCommon.UserInput

public protocol ModelProtocol: AsyncSequence where Element == String {
    init(
        directory: URL,
        input: UserInput,
        maxOutputTokenCount: Int?
    ) throws
}

public extension ModelProtocol {
    init(
        directory _: URL,
        input _: UserInput,
        maxOutputTokenCount _: Int?
    ) throws {
        throw SHLLMError.unimplemented
    }

    var result: String {
        get async throws {
            var result = ""
            for try await chunk in self {
                result += chunk
            }
            return result
        }
    }

    func toolResult<T: Codable>() async throws -> T {
        let decoder = JSONDecoder()
        let result = try await result.toolCall()
        return try decoder.decode(T.self, from: Data(result.utf8))
    }
}

private extension String {
    func toolCall() -> String {
        let prefix = "<tool_call>"
        let suffix = "</tool_call>"

        let startIndex = ranges(of: prefix).last?.lowerBound ?? startIndex
        let endIndex = ranges(of: suffix).last?.upperBound ?? endIndex

        let whitespace = CharacterSet.whitespacesAndNewlines
        var result = self[startIndex ..< endIndex]
            .trimmingCharacters(in: whitespace)

        if result.hasPrefix(prefix) {
            result.removeFirst(prefix.count)
        }

        if result.hasSuffix(suffix) {
            result.removeLast(suffix.count)
        }

        return result.trimmingCharacters(in: whitespace)
    }
}
