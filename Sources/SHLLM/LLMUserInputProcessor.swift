import Foundation
import MLX
import MLXLMCommon
import Tokenizers

struct LLMUserInputProcessor: UserInputProcessor {
    private let tokenizer: Tokenizer
    private let configuration: ModelConfiguration

    init(tokenizer: any Tokenizer, configuration: ModelConfiguration) {
        self.tokenizer = tokenizer
        self.configuration = configuration
    }

    func prepare(input: UserInput) throws -> LMInput {
        do {
            let messages = input.prompt.asMessages()
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools,
                additionalContext: input.additionalContext
            )
            return LMInput(tokens: MLXArray(promptTokens))
        } catch {
            // #150 -- it might be a TokenizerError.chatTemplate("No chat template was
            // specified")
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
