import Foundation

public actor CodeLlama: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        try LLM.assertSupportedDevice
        let llm = try await LLM.llama(directory: directory)
        self.llm = .init(llm)
    }
}

extension CodeLlama {
    static var bundleDirectory: URL {
        get throws {
            let dir = "CodeLlama-13b-Instruct-hf-4bit-MLX"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}
