import Foundation
import MLX
import MLXLMCommon
import Tokenizers

struct LLMUserInputProcessor: UserInputProcessor {
    private let tokenizer: Tokenizer
    private let configuration: ModelConfiguration
    private let messageGenerator: MessageGenerator
    private let maxInputTokenCount: Int?

    init(
        tokenizer: any Tokenizer,
        configuration: ModelConfiguration,
        messageGenerator: MessageGenerator,
        maxInputTokenCount: Int? = nil
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.messageGenerator = messageGenerator
        self.maxInputTokenCount = maxInputTokenCount
    }

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        do {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages,
                chatTemplate: nil,
                addGenerationPrompt: true,
                truncation: maxInputTokenCount != nil,
                maxLength: maxInputTokenCount,
                tools: input.tools,
                additionalContext: input.additionalContext
            )
            return LMInput(tokens: MLXArray(promptTokens))
        } catch TokenizerError.missingChatTemplate {
            print(
                "No chat template was included or provided, so converting messages to simple text format. This is not optimal for model performance, so applications should provide a chat template if none is included with the model."
            )
            let prompt =
                messages
                    .compactMap { $0["content"] as? String }
                    .joined(separator: "\n\n")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}
