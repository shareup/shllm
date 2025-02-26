import Foundation

public actor Qwen2_5__1_5B: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        try LLM.assertSupportedDevice
        let llm = try await LLM.qwen2(directory: directory)
        self.llm = .init(llm)
    }
}

extension Qwen2_5__1_5B {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Qwen2.5-1.5B-Instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}
