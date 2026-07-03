import MLXLMCommon
import MLXVLM
@testable import SHLLM
import Testing

@Suite
struct Gemma4ChannelParserTests {
    @Test
    func separatesReasoningFromText() {
        let output = parse(chunks: [
            "<|channel>",
            "thought",
            "\n",
            "I need to think about this.",
            "<channel|>",
            "Here is my answer.",
        ])

        #expect(output.reasoning == "I need to think about this.")
        #expect(output.text == "Here is my answer.")
        #expect(output.toolCalls.isEmpty)
    }

    @Test
    func handlesEmptyThinkingBlock() {
        let output = parse(chunks: [
            "<|channel>",
            "thought",
            "\n",
            "<channel|>",
            "The price of AAPL is $123.45.",
        ])

        #expect(output.reasoning.isEmpty)
        #expect(output.text == "The price of AAPL is $123.45.")
        #expect(output.toolCalls.isEmpty)
    }

    @Test
    func passesPlainTextThroughWhenNoChannelHeader() {
        let output = parse(chunks: [
            "Just a plain response without channel markers.",
        ])

        #expect(output.reasoning.isEmpty)
        #expect(output.text == "Just a plain response without channel markers.")
        #expect(output.toolCalls.isEmpty)
    }

    @Test
    func emitsBothResponsesWhenEndTagIsMergedWithFinalText() {
        let output = parse(chunks: [
            "<|channel>thought\nReasoning here.<channel|>Final answer.",
        ])

        #expect(output.reasoning == "Reasoning here.")
        #expect(output.text == "Final answer.")
        #expect(output.toolCalls.isEmpty)
    }

    @Test
    func handlesHeaderSplitAcrossChunks() {
        let output = parse(chunks: [
            "<|channel>thought",
            "\n",
            "Some reasoning.",
            "<channel|>",
            "The answer.",
        ])

        #expect(output.reasoning == "Some reasoning.")
        #expect(output.text == "The answer.")
        #expect(output.toolCalls.isEmpty)
    }

    @Test
    func handlesEndTagSplitAcrossChunks() {
        let output = parse(chunks: [
            "<|channel>thought\n",
            "Reasoning",
            "<chan",
            "nel|>",
            "Text after split.",
        ])

        #expect(output.reasoning == "Reasoning")
        #expect(output.text == "Text after split.")
        #expect(output.toolCalls.isEmpty)
    }

    @Test
    func consumesOnlyTheHeaderNewline() {
        let output = parse(chunks: [
            "<|channel>thought\n\nReasoning starts after a blank line.",
            "<channel|>",
            "Done.",
        ])

        #expect(output.reasoning == "\nReasoning starts after a blank line.")
        #expect(output.text == "Done.")
        #expect(output.toolCalls.isEmpty)
    }

    @Test
    func handlesCharacterByCharacterMarkers() {
        let chunks = Array("<|channel>thought\nThink.<channel|>Answer.").map(String.init)
        let output = parse(chunks: chunks)

        #expect(output.reasoning == "Think.")
        #expect(output.text == "Answer.")
        #expect(output.toolCalls.isEmpty)
    }

    @Test
    func passesToolCallsThrough() {
        let toolCall = ToolCall(
            function: ToolCall.Function(
                name: "get_stock_price",
                arguments: ["symbol": "AAPL"]
            )
        )
        let output = parse(generations: [
            .chunk("<|channel>thought\n<channel|>"),
            .toolCall(toolCall),
            .info(.test),
        ])

        #expect(output.reasoning.isEmpty)
        #expect(output.text.isEmpty)
        #expect(output.toolCalls == [toolCall])
    }

    @Test
    func flushesQueuedFinalTextOnInfo() {
        let output = parse(generations: [
            .chunk("<|channel>thought\nReasoning.<channel|>Final."),
            .info(.test),
        ])

        #expect(output.reasoning == "Reasoning.")
        #expect(output.text == "Final.")
        #expect(output.toolCalls.isEmpty)
    }

    @Test
    func flushesPartialEndTagOnInfo() {
        let output = parse(generations: [
            .chunk("<|channel>thought\nReasoning<chan"),
            .info(.test),
        ])

        #expect(output.reasoning == "Reasoning<chan")
        #expect(output.text.isEmpty)
        #expect(output.toolCalls.isEmpty)
    }

    @Test
    func flushesPartialHeaderOnInfo() {
        let output = parse(generations: [
            .chunk("<|channel>thought"),
            .info(.test),
        ])

        #expect(output.reasoning.isEmpty)
        #expect(output.text == "<|channel>thought")
        #expect(output.toolCalls.isEmpty)
    }

    private func parse(chunks: [String]) -> ParsedOutput {
        parse(generations: chunks.map(Generation.chunk) + [.info(.test)])
    }

    private func parse(generations: [Generation]) -> ParsedOutput {
        let parser = LLM<Gemma4Unified>.gemma4Parser
        var output = ParsedOutput()

        for generation in generations {
            guard let response = parser.parse(generation) else {
                continue
            }

            switch response {
            case let .reasoning(delta):
                output.reasoning += delta

            case let .text(delta):
                output.text += delta

            case let .toolCall(toolCall):
                output.toolCalls.append(toolCall)
            }
        }

        return output
    }

    private struct ParsedOutput {
        var reasoning = ""
        var text = ""
        var toolCalls = [ToolCall]()
    }
}

private extension GenerateCompletionInfo {
    static var test: GenerateCompletionInfo {
        GenerateCompletionInfo(
            promptTokenCount: 0,
            generationTokenCount: 0,
            promptTime: 1,
            generationTime: 1
        )
    }
}
