import Foundation

public actor Qwen1_5: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        try LLM.assertSupportedDevice
        let llm = try await LLM.qwen2(directory: directory)
        self.llm = .init(llm)
    }
}

extension Qwen1_5 {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Qwen1.5-0.5B-Chat-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}
