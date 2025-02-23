import Foundation
import struct Hub.Config
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

public actor Mistral7B: ModelProtocol {
    public let llm: AsyncLockedValue<LLM>

    public init(directory: URL) async throws {
        let llm = try await LLM.llama(directory: directory)
        self.llm = .init(llm)
    }
}

extension Mistral7B {
    static var bundleDirectory: URL {
        get throws {
            let dir = "Mistral-7B-Instruct-v0.3-4bit"
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
