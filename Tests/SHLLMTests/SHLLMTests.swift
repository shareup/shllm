@testable import SHLLM
import Testing

@Test
func canBuildLibrary() async throws {
    #expect(Bool(true))
}

@Test
func cacheLimit() async throws {
    let limit = 1024
    #expect(SHLLM.cacheLimit != limit)
    SHLLM.cacheLimit = limit
    #expect(SHLLM.cacheLimit == limit)
}

@Test
func memoryLimit() async throws {
    let limit = 1024
    #expect(SHLLM.memoryLimit != limit)
    SHLLM.memoryLimit = limit
    #expect(SHLLM.memoryLimit == limit)
}

@Test
func recommendedMaxWorkingSetSize() async throws {
    let recommended = SHLLM.recommendedMaxWorkingSetSize
    #expect(recommended > 0)
}

// NOTE: Running inference on the CPU takes way too long.
@Test(.enabled(if: false))
func onCPU() async throws {
    guard SHLLM.isSupportedDevice else {
        Swift.print("⚠️ Metal GPU not available")
        return
    }

    let input: UserInput = .init(messages: [
        ["role": "system", "content": "You are a helpful assistant."],
        ["role": "user", "content": "What is the meaning of life?"],
    ])

    try await SHLLM.withDefaultDevice(.cpu) {
        guard let llm = try loadModel(
            directory: LLM.gemma3_1B,
            input: input,
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<end_of_turn>"]
                return config
            }
        ) as LLM<Gemma3TextModel>? else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(!response.isEmpty)
    }
}

@Test()
func onGPU() async throws {
    guard SHLLM.isSupportedDevice else {
        Swift.print("⚠️ Metal GPU not available")
        return
    }

    let input: UserInput = .init(messages: [
        ["role": "system", "content": "You are a helpful assistant."],
        ["role": "user", "content": "What is the meaning of life?"],
    ])

    try await SHLLM.withDefaultDevice(.gpu) {
        guard let llm = try loadModel(
            directory: LLM.gemma3_1B,
            input: input,
            customConfiguration: { config in
                var config = config
                config.extraEOSTokens = ["<end_of_turn>"]
                return config
            }
        ) as LLM<Gemma3TextModel>? else { return }

        var response = ""
        for try await token in llm.text {
            response += token
        }

        Swift.print(response)
        #expect(!response.isEmpty)
    }
}
