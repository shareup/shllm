import Foundation

public actor OpenELM: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        let llm = try await LLM.openELM(directory: directory)
        self.llm = .init(llm)
    }
}

extension OpenELM {
    static var bundleDirectory: URL {
        get throws {
            let dir = "OpenELM-270M-Instruct"
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
