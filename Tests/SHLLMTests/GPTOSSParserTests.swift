import MLXLLM
import MLXLMCommon
@testable import SHLLM
import Testing

@Suite
struct GPTOSSParserTests {
    @Test
    func streamsAssistantCommentaryPreambleAsText() throws {
        let parser = LLM<GPTOSSModel>.gptOSSParser

        // Simulate a Harmony assistant commentary message without a recipient:
        // <|start|>assistant<|channel|>commentary<|message|>Preamble<|end|>
        // Note: StreamableParser is initialized with startingRole = .assistant,
        // so we do not need to send the <|start|> token.
        let tokens = [
            "assistant",
            "<|channel|>",
            "commentary",
            "<|message|>",
            "Action plan: prepare environment.",
        ]

        var text = ""
        var sawReasoning = false
        var sawToolCall = false

        for tok in tokens {
            if let response = parser.parse(.chunk(tok)) {
                switch response {
                case let .text(delta):
                    text += delta
                case .reasoning:
                    sawReasoning = true
                case .toolCall:
                    sawToolCall = true
                }
            }
        }

        // Close the message
        _ = parser.parse(.chunk("<|end|>"))

        #expect(!text.isEmpty)
        #expect(text.contains("Action plan:"))
        #expect(!sawReasoning)
        #expect(!sawToolCall)
    }
}
