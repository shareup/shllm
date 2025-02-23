@testable import SHLLM
import Testing

private extension Qwen1_5 {
    init() async throws {
        try await self.init(directory: Self.bundleDirectory)
    }
}

@Test
func canLoadAndQueryQwen1_5() async throws {
    let llm = try await Qwen1_5()
    let result = try await llm.request(.init(messages: [
        ["role": "system", "content": "You are a helpful assistant."],
        ["role": "user", "content": "What is the meaning of life?"],
    ]))
    Swift.print(result)
    #expect(!result.isEmpty)
}
