@testable import SHLLM
import Testing

private extension Mistral7B {
    init() async throws {
        try await self.init(directory: Self.bundleDirectory)
    }
}

@Test
func canLoadAndQueryMistral7B() async throws {
    let llm = try await Mistral7B()
    let result = try await llm.request(.init(messages: [
        ["role": "system", "content": "You are a helpful assistant."],
        ["role": "user", "content": "What is the meaning of life?"],
    ]))
    Swift.print(result)
    #expect(!result.isEmpty)
}
