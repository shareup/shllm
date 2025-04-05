import Foundation
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Gemma2_9BTests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try gemma2_9B(input) else { return }

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

        guard let llm = try gemma2_9B(input) else { return }

        let result = try await llm.result
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}

private func gemma2_9B(
    _ input: UserInput
) throws -> LLM<Gemma2Configuration, Gemma2Model>? {
    try loadModel(
        LLM.gemma2_9B,
        directory: LLM.gemma2_9B,
        input: input
    )
}
