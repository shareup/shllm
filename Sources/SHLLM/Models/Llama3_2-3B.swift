import Foundation

public actor Llama3_2__3B: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        try LLM.assertSupportedDevice
        let llm = try await LLM.llama(directory: directory)
        self.llm = .init(llm)
    }
}

extension Llama3_2__3B {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Llama-3.2-3B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}
