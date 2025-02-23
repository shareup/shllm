import Foundation

public actor Qwen2_5__7B: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        let llm = try await LLM.qwen2(directory: directory)
        self.llm = .init(llm)
    }
}

extension Qwen2_5__7B {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Qwen2.5-7B-Instruct-4bit"
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
