import Foundation
@testable import SHLLM
import Testing

@Suite(.serialized)
struct CodeLlamaTests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful JavaScript coding assistant."],
            [
                "role": "user",
                "content": "Express the meaning of life in a JavaScript function.",
            ],
        ])

        guard let llm = try codeLlama(input) else { return }

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
            ["role": "system", "content": "You are a helpful JavaScript coding assistant."],
            [
                "role": "user",
                "content": "Express the meaning of life in a JavaScript function.",
            ],
        ])

        guard let llm = try codeLlama(input) else { return }

        let result = try await llm.result
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}

private func codeLlama(
    _ input: UserInput
) throws -> LLM<LlamaModel>? {
    try loadModel(
        directory: LLM.codeLlama,
        input: input
    )
}
