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
        var hasReasoning = false
        var hasToolCall = false

        for token in tokens {
            if let response = parser.parse(.chunk(token)) {
                switch response {
                case let .text(delta):
                    text += delta
                case .reasoning:
                    hasReasoning = true
                case .toolCall:
                    hasToolCall = true
                }
            }
        }

        // Close the message
        _ = parser.parse(.chunk("<|end|>"))

        #expect(!text.isEmpty)
        #expect(text.contains("Action plan:"))
        #expect(!hasReasoning)
        #expect(!hasToolCall)
    }
}
