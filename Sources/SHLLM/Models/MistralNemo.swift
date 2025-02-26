import Foundation

public actor MistralNemo: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        try LLM.assertSupportedDevice
        let llm = try await LLM.llama(directory: directory)
        self.llm = .init(llm)
    }
}

extension MistralNemo {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Mistral-Nemo-Instruct-2407-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}
