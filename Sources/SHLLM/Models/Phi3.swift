import Foundation
import struct Hub.Config
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

public actor Phi3 {
    private let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        let llm = try await LLM.phi3(directory: directory)
        self.llm = .init(llm)
    }

    public func request(
        _ input: UserInput,
        maxTokenCount: Int = 1024 * 1024
    ) async throws -> String {
        try await llm.withLock { llm in
            try await llm.request(input, maxTokenCount: maxTokenCount)
        }
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
