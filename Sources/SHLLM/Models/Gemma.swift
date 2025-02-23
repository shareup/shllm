import Foundation

public actor Gemma: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        let llm = try await LLM.gemma(directory: directory)
        self.llm = .init(llm)
    }
}

extension Gemma {
    static var bundleDirectory: URL {
        get throws {
            let dir = "quantized-gemma-2b-it"
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
