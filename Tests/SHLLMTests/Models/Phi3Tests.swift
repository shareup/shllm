import Foundation
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Phi3Tests {
    @Test
    func canStreamResult() async throws {
        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try phi3(input) else { return }

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

        guard let llm = try phi3(input) else { return }

        let result = try await llm.text.result
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}

private func phi3(
    _ input: UserInput
) throws -> LLM<Phi3Model>? {
    try loadModel(
        directory: LLM.phi3,
        input: input
    )
}
