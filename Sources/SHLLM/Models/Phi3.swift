import Foundation

public actor Phi3: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        let llm = try await LLM.phi3(directory: directory)
        self.llm = .init(llm)
    }
}

extension Phi3 {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Phi-3.5-mini-instruct-4bit"
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
