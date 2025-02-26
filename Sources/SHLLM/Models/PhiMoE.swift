import Foundation

public actor PhiMoE: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        try LLM.assertSupportedDevice
        let llm = try await LLM.phiMoE(directory: directory)
        self.llm = .init(llm)
    }
}

extension PhiMoE {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Phi-3.5-MoE-instruct-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}
