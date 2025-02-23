import Foundation
import struct Hub.Config
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

public actor MistralNemo {
    private let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        let llm = try await LLM.mistralNemo(directory: directory)
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

extension MistralNemo {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Mistral-Nemo-Instruct-2407-4bit"
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
