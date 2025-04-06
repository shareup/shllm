import Foundation
@testable import SHLLM
import Testing

@Suite(.serialized)
struct MistralNemoTests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try mistralNemo(input) else { return }

        var result = ""
        for try await reply in llm {
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

        guard let llm = try mistralNemo(input) else { return }

        let result = try await llm.result
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}

private func mistralNemo(
    _ input: UserInput
) throws -> LLM<LlamaConfiguration, LlamaModel>? {
    try loadModel(
        LLM.mistralNemo,
        directory: LLM.mistralNemo,
        input: input
    )
}
