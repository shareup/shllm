import Foundation

public actor Mistral7B: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        try LLM.assertSupportedDevice
        let llm = try await LLM.llama(directory: directory)
        self.llm = .init(llm)
    }
}

extension Mistral7B {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Mistral-7B-Instruct-v0.3-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}
