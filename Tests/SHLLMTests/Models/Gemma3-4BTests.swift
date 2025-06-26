import Foundation
import MLXVLM
@testable import SHLLM
import Testing

@Suite(.serialized)
struct Gemma3_4BTests {
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

        guard let llm = try gemma3_4B(input: input) else { return }

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

        guard let llm = try gemma3_4B(input: input) else { return }

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
        guard let llm = try gemma3_4B(image: data) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(response.contains("The 3 authentication factors"))
        #expect(response.contains("Something you forgot"))
        #expect(response.contains("Something you left in the taxi"))
        #expect(response.contains("Something that can be chopped off"))
    }

    @Test()
    @MainActor
    func canExtractTextFromImageURL() async throws {
        guard SHLLM.isSupportedDevice else {
            Swift.print("⚠️ Metal GPU not available")
            return
        }

        let url = try authenticationFactorsURL
        guard let llm = try gemma3_4B(image: url) else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(response.contains("The 3 authentication factors"))
        #expect(response.contains("Something you forgot"))
        #expect(response.contains("Something you left in the taxi"))
        #expect(response.contains("Something that can be chopped off"))
    }
}

private extension Gemma3_4BTests {
    func gemma3_4B(
        input: UserInput
    ) throws -> LLM<Gemma3>? {
        try loadModel(
            directory: LLM.gemma3_4B,
            input: input,
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<end_of_turn>"]
                return config
            }
        )
    }

    func gemma3_4B(
        image: Data
    ) throws -> LLM<Gemma3>? {
        try loadModel(
            directory: LLM.gemma3_4B,
            input: imageInput(image),
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<end_of_turn>"]
                return config
            }
        )
    }

    func gemma3_4B(
        image: URL
    ) throws -> LLM<Gemma3>? {
        try loadModel(
            directory: LLM.gemma3_4B,
            input: imageInput(image),
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<end_of_turn>"]
                return config
            }
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
