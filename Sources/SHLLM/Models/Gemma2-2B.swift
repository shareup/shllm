import Foundation

public actor Gemma2_2B: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        try LLM.assertSupportedDevice
        let llm = try await LLM.gemma2(directory: directory)
        self.llm = .init(llm)
    }
}

extension Gemma2_2B {
    static var bundleDirectory: URL {
        get throws {
            let dir = "gemma-2-2b-it-4bit"
            return try Bundle.shllm.directory(named: dir)
        }
    }
}
