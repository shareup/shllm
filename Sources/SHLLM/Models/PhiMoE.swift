import Foundation

public actor PhiMoE: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        let llm = try await LLM.phiMoE(directory: directory)
        self.llm = .init(llm)
    }
}

extension PhiMoE {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Phi-3.5-MoE-instruct-4bit"
            guard let url = Bundle.shllm.url(
                forResource: dir,
                withExtension: nil,
                subdirectory: "Resources"
            ) else {
                throw SHLLMError.directoryNotFound(dir)
            }
            return url
        }
    }
}
