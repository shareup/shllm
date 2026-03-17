import Foundation
import MLXLMCommon
import Synchronized

enum ThinkingTagProcessor<Model: LanguageModel> {
    /// Creates a `LLM.ResponseParser` that processes thinking tags
    /// in the model's output. It assumes the model will output
    /// start and end tags to delimit reasoning blocks. This means
    /// the parser assumes arriving tokens are non-reasoning tokens
    /// unless they are between start and end tags.
    static func hybrid(
        startTags: Set<String> = ["<think>", "<think>\n"],
        endTags: Set<String> = ["</think>", "</think>\n"]
    ) -> LLM<Model>.ResponseParser {
        parser(
            mode: .hybrid,
            startTags: startTags,
            endTags: endTags
        )
    }

    /// Creates a `LLM.ResponseParser` that processes thinking tags
    /// in the model's output. It assumes the model starts in
    /// "thinking" mode, meaning arriving tokens are assumed to
    /// be reasoning tokens until an end tag is seen.
    static func defaultsToThinking(
        startTags: Set<String> = ["<think>", "<think>\n"],
        endTags: Set<String> = ["</think>", "</think>\n"]
    ) -> LLM<Model>.ResponseParser {
        parser(
            mode: .defaultsToThinking,
            startTags: startTags,
            endTags: endTags
        )
    }
}

private enum Mode: Sendable {
    case hybrid
    case defaultsToThinking
}

private func parser<Model: LanguageModel>(
    mode: Mode,
    startTags: Set<String>,
    endTags: Set<String>
) -> LLM<Model>.ResponseParser {
    let isThinking = Locked(mode == .defaultsToThinking)
    return LLM.ResponseParser { generation in
        isThinking.access { thinking in
            switch generation {
            case let .chunk(chunk):
                if startTags.contains(chunk) {
                    thinking = true
                    return nil
                } else if endTags.contains(chunk) {
                    thinking = false
                    return nil
                } else if thinking {
                    return .reasoning(chunk)
                } else {
                    return .text(chunk)
                }

            case let .toolCall(toolCall):
                thinking = false
                return .toolCall(toolCall)

            case .info:
                return nil
            }
        }
    }
}
