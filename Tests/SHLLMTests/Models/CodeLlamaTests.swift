@testable import SHLLM
import Testing

extension CodeLlama: InitializableWithDirectory {
    static var tests: Self? {
        get async throws {
            try await loadModel(from: bundleDirectory)
        }
    }
}

@Suite(.serialized)
struct CodeLlamaTests {
    @Test
    func canLoadAndQuery() async throws {
        guard let llm = try await CodeLlama.tests else { return }
        let result = try await llm.request(.init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            [
                "role": "user",
                "content": "Express the meaning of life in a JavaScript function.",
            ],
        ]))
        Swift.print(result)
        #expect(!result.isEmpty)
    }
}
