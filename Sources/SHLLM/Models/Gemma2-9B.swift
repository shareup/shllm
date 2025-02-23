import Foundation

public actor Gemma2_9B: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        let llm = try await LLM.gemma2(directory: directory)
        self.llm = .init(llm)
    }
}

extension Gemma2_9B {
    static var bundleDirectory: URL {
        get throws {
            let dir = "gemma-2-9b-it-4bit"
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
