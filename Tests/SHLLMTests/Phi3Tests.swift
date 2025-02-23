@testable import SHLLM
import Testing

extension Phi3 {
    init() async throws {
        try await self.init(directory: Self.bundleDirectory)
    }
}

@Test
func canLoadAndQueryModel() async throws {
    let phi3 = try await Phi3()
    let result = try await phi3.request(.init(messages: [
        ["role": "system", "content": "You are a helpful assistant."],
        ["role": "user", "content": "What is the meaning of life?"],
    ]))
    Swift.print(result)
    #expect(!result.isEmpty)
}
