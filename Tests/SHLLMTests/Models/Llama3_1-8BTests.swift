import Foundation
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Llama3_1__8BTests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try llama3_1__8B(input) else { return }

        var result = ""
        for try await reply in llm.text {
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

        guard let llm = try llama3_1__8B(input) else { return }

        let result = try await llm.text.result
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}

private func llama3_1__8B(
    _ input: UserInput
) throws -> LLM<LlamaModel>? {
    try loadModel(
        directory: LLM.llama3_1__8B,
        input: input
    )
}
