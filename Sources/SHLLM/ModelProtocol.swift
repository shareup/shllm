import Foundation
import MLXLMCommon

public protocol ModelProtocol {
    var llm: AsyncLockedValue<LLM> { get async }
}

public extension ModelProtocol {
    func request<T: Codable>(
        tools: Tools,
        messages: [Message],
        maxTokenCount: Int = 1024 * 1024
    ) async throws -> T {
        try await llm.withLock { llm in
            try await llm.request(tools: tools, messages: messages, maxTokenCount: maxTokenCount)
        }
    }

    func request(
        messages: [Message],
        maxTokenCount: Int = 1024 * 1024
    ) async throws -> String {
        try await llm.withLock { llm in
            try await llm.request(messages: messages, maxTokenCount: maxTokenCount)
        }
    }

    func request(
        _ input: UserInput,
        maxTokenCount: Int = 1024 * 1024
    ) async throws -> String {
        try await llm.withLock { llm in
            try await llm.request(input, maxTokenCount: maxTokenCount)
        }
    }
}
