import Hub
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

public struct SHLLM {
    public init() {
        Swift.print(Bundle.shllm.url(
            forResource: "a",
            withExtension: "png",
            subdirectory: "Resources"
        )?.description ?? "")
    }

    public func hello() async throws -> Void {
        let url = Bundle.shllm.url(
            forResource: "Qwen2.5-1.5B-Instruct-4bit",
            withExtension: nil,
            subdirectory: "Resources"
        )!

        let configFileUrl = Bundle.shllm.url(
            forResource: "config",
            withExtension: "json",
            subdirectory: "Resources/Qwen2.5-1.5B-Instruct-4bit"
        )!

        let tokenizerConfigFileUrl = Bundle.shllm.url(
            forResource: "tokenizer_config",
            withExtension: "json",
            subdirectory: "Resources/Qwen2.5-1.5B-Instruct-4bit"
        )!

        let tokenizerDataFileUrl = Bundle.shllm.url(
            forResource: "tokenizer",
            withExtension: "json",
            subdirectory: "Resources/Qwen2.5-1.5B-Instruct-4bit"
        )!

        let configData = try Data(contentsOf: configFileUrl)
        let config = try JSONDecoder().decode(Qwen2Configuration.self, from: configData)

        let tokenizerConfig = Config(try JSONSerialization.jsonObject(with: Data(contentsOf: tokenizerConfigFileUrl)) as! [NSString: Any])
        let tokenizerData = Config(try JSONSerialization.jsonObject(with: Data(contentsOf: tokenizerDataFileUrl)) as! [NSString: Any])

        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        Swift.print(tokenizer.encode(text: "Hello world").description)

        let configuration = ModelConfiguration(directory: url, overrideTokenizer: nil, defaultPrompt: "Why is the sky blue?")
        let inputProcessor = LLMUserInputProcessor(tokenizer: tokenizer, configuration: configuration)

        let langModel = Qwen2Model(config)
        let context = ModelContext(configuration: configuration, model: langModel, processor: inputProcessor, tokenizer: tokenizer)
        let container = ModelContainer(context: context)
        let generateParameters = MLXLMCommon.GenerateParameters(temperature: 0.6)

        let result = try await container.perform { ctx in
            let input = try await ctx.processor.prepare(input: .init(prompt: "Why is Berlin famous?"))

            var detokenizer = NaiveStreamingDetokenizer(tokenizer: ctx.tokenizer)

            return try MLXLMCommon.generate(input: input, parameters: generateParameters, context: ctx) { tokens in
                for token in tokens {
                    detokenizer.append(token: token)
                }

                if tokens.count >= 150 {
                    return .stop
                } else {
                    return .more
                }
            }
        }

        Swift.print(result)
    }
}

private struct LLMUserInputProcessor: UserInputProcessor {
    let tokenizer: Tokenizer
    let configuration: ModelConfiguration

    internal init(tokenizer: any Tokenizer, configuration: ModelConfiguration) {
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    func prepare(input: UserInput) throws -> LMInput {
        do {
            let messages = input.prompt.asMessages()
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)
            return LMInput(tokens: MLXArray(promptTokens))
        } catch {
            // #150 -- it might be a TokenizerError.chatTemplate("No chat template was specified")
            // but that is not public so just fall back to text
            let prompt = input.prompt
                .asMessages()
                .compactMap { $0["content"] as? String }
                .joined(separator: ". ")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}

private extension Bundle {
    // This is nearly identical to the generated code of `Bundle.module`. However,
    // `Bundle.module` does not work correctly in tests. There is a thread on the
    // Swift Forums describing the issue:
    // https://forums.swift.org/t/swift-5-3-spm-resources-in-tests-uses-wrong-bundle-path/37051
    static var shllm: Bundle = {
        let bundleName = "SHLLM_SHLLM"

        var candidates = [
            // Bundle should be present here when the package is linked into an App.
            Bundle.main.resourceURL,

            // Bundle should be present here when the package is linked into a framework.
            Bundle(for: BundleLocator.self).resourceURL,

            // For command-line tools.
            Bundle.main.bundleURL,

            // iOS Xcode previews.
            Bundle(for: BundleLocator.self)
                .resourceURL?
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent(),

            // macOS Xcode previews.
            Bundle(for: BundleLocator.self)
                .resourceURL?
                .deletingLastPathComponent()
                .deletingLastPathComponent(),
        ]

        // For tests
        // https://forums.swift.org/t/swift-5-3-spm-resources-in-tests-uses-wrong-bundle-path/37051/21
        candidates += Bundle.allBundles.compactMap(\.resourceURL)

        for candidate in candidates {
            let bundlePath = candidate?.appendingPathComponent(bundleName + ".bundle")
            if let bundle = bundlePath.flatMap(Bundle.init(url:)) {
                return bundle
            }
        }
        fatalError("unable to find bundle named '\(bundleName)'")
    }()
}

private class BundleLocator: NSObject {}
