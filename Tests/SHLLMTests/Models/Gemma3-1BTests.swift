import Foundation
import MLXVLM
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Gemma3_1BTests {
    @Test
    func canStreamResult() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try gemma3_1B(input: input) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(!response.isEmpty)
    }

    @Test
    func canAwaitResult() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try gemma3_1B(input: input) else { return }

        let response = try await llm.text.result

        Swift.print(response)
        #expect(!response.isEmpty)
    }
}

private extension Gemma3_1BTests {
    func gemma3_1B(
        input: UserInput
    ) throws -> LLM<Gemma3TextModel>? {
        try loadModel(
            directory: LLM.gemma3_1B,
            input: input,
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<end_of_turn>"]
                return config
            }
        )
    }
}
