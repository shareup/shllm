import Combine
import Foundation
import MLXLMCommon

public typealias Message = MLXLMCommon.Message
public typealias UserInput = MLXLMCommon.UserInput

public protocol ModelProtocol: AsyncSequence where Element == String {
    init(
        directory: URL,
        input: UserInput,
        maxInputTokenCount: Int?,
        maxOutputTokenCount: Int?
    ) throws
}

extension LLM: ModelProtocol {}

public extension ModelProtocol {
    init(
        directory _: URL,
        input _: UserInput,
        maxInputTokenCount _: Int?,
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
        let result = try await result.trimmingToolCallMarkup()
        return try decoder.decode(T.self, from: Data(result.utf8))
    }
}

private extension String {
    func trimmingToolCallMarkup() -> String {
        let prefix = "<tool_call>"
        let suffix = "</tool_call>"

        let whitespace = CharacterSet.whitespacesAndNewlines
        var copy = trimmingCharacters(in: whitespace)
        copy.removeFirst(prefix.count)
        copy.removeLast(suffix.count)
        return copy.trimmingCharacters(in: whitespace)
    }
}
