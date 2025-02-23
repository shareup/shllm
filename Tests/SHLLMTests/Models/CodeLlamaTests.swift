@testable import SHLLM
import Testing

private extension CodeLlama {
    init() async throws {
        try await self.init(directory: Self.bundleDirectory)
    }
}

@Test
func canLoadAndQueryCodeLlama() async throws {
    let llm = try await CodeLlama()
    let result = try await llm.request(.init(messages: [
        ["role": "system", "content": "You are a helpful assistant."],
        ["role": "user", "content": "Express the meaning of life in a JavaScript function."],
    ]))
    Swift.print(result)
    #expect(!result.isEmpty)
}
