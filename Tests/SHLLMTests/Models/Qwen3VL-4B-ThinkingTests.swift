import Foundation
import MLXVLM
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Qwen3VL_4B_ThinkingTests {
    @Test
    func canStreamResult() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try qwen3VL_4B(input: input) else { return }

        var reasoning = ""
        var response = ""
        for try await reply in llm {
            switch reply {
            case let .reasoning(text):
                reasoning.append(text)
            case let .text(text):
                response.append(text)
            case .toolCall:
                Issue.record()
            }
        }

        Swift.print("<think>\n\(reasoning)\n</think>")
        Swift.print(response)
        #expect(!response.isEmpty)
    }

    @Test
    func canAwaitResult() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let input: UserInput = .init(messages: [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"],
        ])

        guard let llm = try qwen3VL_4B(input: input) else { return }

        let response = try await llm.text.result

        Swift.print(response)
        #expect(!response.isEmpty)
    }

    @Test()
    @MainActor
    func canExtractTextFromImageData() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let data = try authenticationFactors
        guard let llm = try qwen3VL_4B(image: data) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(response.contains("authentication"))
    }

    @Test()
    @MainActor
    func canExtractTextFromImageURL() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let url = try authenticationFactorsURL
        guard let llm = try qwen3VL_4B(image: url) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(response.contains("authentication"))
    }
}

private extension Qwen3VL_4B_ThinkingTests {
    func qwen3VL_4B(
        input: UserInput
    ) throws -> LLM<Qwen3VL>? {
        try loadModel(
            directory: LLM.qwen3VL_4B_Thinking,
            input: input,
            responseParser: LLM<Qwen3VL>.qwen3VLThinkingParser
        )
    }

    func qwen3VL_4B(
        image: Data
    ) throws -> LLM<Qwen3VL>? {
        try loadModel(
            directory: LLM.qwen3VL_4B_Thinking,
            input: imageInput(image),
            responseParser: LLM<Qwen3VL>.qwen3VLThinkingParser
        )
    }

    func qwen3VL_4B(
        image: URL
    ) throws -> LLM<Qwen3VL>? {
        try loadModel(
            directory: LLM.qwen3VL_4B_Thinking,
            input: imageInput(image),
            responseParser: LLM<Qwen3VL>.qwen3VLThinkingParser
        )
    }

    var authenticationFactorsURL: URL {
        get throws {
            guard let url = Bundle.module.url(
                forResource: "3-authentication-factors",
                withExtension: "png"
            ) else {
                throw NSError(
                    domain: NSURLErrorDomain,
                    code: NSURLErrorFileDoesNotExist,
                    userInfo: nil
                )
            }
            return url
        }
    }

    var authenticationFactors: Data {
        get throws {
            try Data(contentsOf: authenticationFactorsURL)
        }
    }
}
