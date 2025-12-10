import Foundation
import MLXVLM
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Qwen3VL_4B_InstructTests {
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

        guard let llm = try qwen3VL_4B_Instruct(input: input) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

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

        guard let llm = try qwen3VL_4B_Instruct(input: input) else { return }

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
        guard let llm = try qwen3VL_4B_Instruct(image: data) else { return }

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
        guard let llm = try qwen3VL_4B_Instruct(image: url) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(response.contains("authentication"))
    }
}

private extension Qwen3VL_4B_InstructTests {
    func qwen3VL_4B_Instruct(
        input: UserInput
    ) throws -> LLM<Qwen3VL>? {
        try loadModel(
            directory: LLM.qwen3VL_4B_Instruct,
            input: input,
            responseParser: LLM<Qwen3VL>.qwen3VLInstructParser
        )
    }

    func qwen3VL_4B_Instruct(
        image: Data
    ) throws -> LLM<Qwen3VL>? {
        try loadModel(
            directory: LLM.qwen3VL_4B_Instruct,
            input: imageInput(image),
            responseParser: LLM<Qwen3VL>.qwen3VLInstructParser
        )
    }

    func qwen3VL_4B_Instruct(
        image: URL
    ) throws -> LLM<Qwen3VL>? {
        try loadModel(
            directory: LLM.qwen3VL_4B_Instruct,
            input: imageInput(image),
            responseParser: LLM<Qwen3VL>.qwen3VLInstructParser
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
