@testable import SHLLM
import Testing

extension Mistral7B: InitializableWithDirectory {
    static var tests: Self? {
        get async throws {
            try await loadModel(from: bundleDirectory)
        }
    }
}

@Suite(.serialized)
struct Mistral7BTests {
    @Test
    func canLoadAndQuery() async throws {
        guard let llm = try await Mistral7B.tests else { return }
        let result = try await llm.request(.init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ]))
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}
