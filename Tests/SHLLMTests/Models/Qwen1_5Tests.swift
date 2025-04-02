@testable import SHLLM
import Testing

extension Qwen1_5: InitializableWithDirectory {
    static var tests: Self? {
        get async throws {
            try await loadModel(from: bundleDirectory)
        }
    }
}

@Suite(.serialized)
struct Qwen1_5Tests {
    @Test
    func canLoadAndQuery() async throws {
        guard let llm = try await Qwen1_5.tests else { return }
        let result = try await llm.request(.init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ]))
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}
