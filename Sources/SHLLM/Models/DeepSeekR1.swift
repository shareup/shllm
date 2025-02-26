import Foundation

public actor DeepSeekR1: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        try LLM.assertSupportedDevice
        let llm = try await LLM.qwen2(directory: directory)
        self.llm = .init(llm)
    }
}

extension DeepSeekR1 {
    static var bundleDirectory: URL {
        get throws {
            let dir = "DeepSeek-R1-Distill-Qwen-7B-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}
