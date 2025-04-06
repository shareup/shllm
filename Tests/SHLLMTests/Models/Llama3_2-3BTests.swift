import Foundation
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Llama3_2__3BTests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try llama3_2__3B(input) else { return }

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

        guard let llm = try llama3_2__3B(input) else { return }

        let result = try await llm.result
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}

private func llama3_2__3B(
    _ input: UserInput
) throws -> LLM<LlamaConfiguration, LlamaModel>? {
    try loadModel(
        LLM.llama3_2__3B,
        directory: LLM.llama3_2__3B,
        input: input
    )
}
