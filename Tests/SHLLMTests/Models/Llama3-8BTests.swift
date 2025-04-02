@testable import SHLLM
import Testing

extension Llama3_8B: InitializableWithDirectory {
    static var tests: Self? {
        get async throws {
            try await loadModel(from: bundleDirectory)
        }
    }
}

@Suite(.serialized) struct Llama3_8BTests {
    @Test
    func canLoadAndQuery() async throws {
        guard let llm = try await Llama3_8B.tests else { return }
        let result = try await llm.request(.init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ]))
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}
