import Foundation

public actor Phi2: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        let llm = try await LLM.phi(directory: directory)
        self.llm = .init(llm)
    }
}

extension Phi2 {
    static var bundleDirectory: URL {
        get throws {
            let dir = "phi-2-hf-4bit-mlx"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}
