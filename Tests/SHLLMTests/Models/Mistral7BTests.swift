import Foundation
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Mistral7BTests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try mistral7B(input) else { return }

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

        guard let llm = try mistral7B(input) else { return }

        let result = try await llm.result
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}

private func mistral7B(
    _ input: UserInput
) throws -> LLM<LlamaConfiguration, LlamaModel>? {
    try loadModel(
        LLM.mistral7B,
        directory: LLM.mistral7B,
        input: input
    )
}
