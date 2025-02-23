@testable import SHLLM
import Testing

private extension Llama3_2__3B {
    init() async throws {
        try await self.init(directory: Self.bundleDirectory)
    }
}

@Test
func canLoadAndQueryLlama3_2__3B() async throws {
    let llm = try await Llama3_2__3B()
    let result = try await llm.request(.init(messages: [
        ["role": "system", "content": "You are a helpful assistant."],
        ["role": "user", "content": "What is the meaning of life?"],
    ]))
    Swift.print(result)
    #expect(!result.isEmpty)
}
