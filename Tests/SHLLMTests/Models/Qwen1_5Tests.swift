import Foundation
import MLXLLM
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Qwen1_5Tests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try qwen1_5(input) else { return }

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

        guard let llm = try qwen1_5(input) else { return }

        let result = try await llm.text.result
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}

private func qwen1_5(
    _ input: UserInput
) throws -> LLM<Qwen2Model>? {
    try loadModel(
        directory: LLM.qwen1_5,
        input: input
    )
}
