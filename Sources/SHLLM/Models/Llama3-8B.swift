import Foundation

public actor Llama3_8B: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        let llm = try await LLM.llama(directory: directory)
        self.llm = .init(llm)
    }
}

extension Llama3_8B {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Meta-Llama-3-8B-Instruct-4bit"
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
