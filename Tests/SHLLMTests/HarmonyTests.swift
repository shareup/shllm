import Foundation
@testable import SHLLM
import Synchronized
import Testing

@Suite
struct HarmonyTests {
    @Test
    func streamableParserCountsMessages() async throws {
        let text = [
            "<|start|>assistant<|message|>Hello<|end|>",
            "<|start|>assistant<|channel|>analysis<|message|>Thinking...<|end|>",
            "<|start|>assistant<|channel|>final<|message|>Done<|return|>",
        ].joined()
        let tokens = tokenizeWithSpecials(text)
        var parser = Harmony.StreamableParser()
        for t in tokens {
            try parser.process(t)
        }
        #expect(parser.messages.count == 3)
    }

    @Test
    func streamableParserToolCallWithConstrainAdjacent() async throws {
        // harmony_tests.rs::test_streamable_parser_tool_call_with_constrain_adjacent
        let text =
            "<|start|>assistant<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{\"latitude\":48.8566,\"longitude\":2.3522}<|call|>"
        let tokens = tokenizeWithSpecials(text)
        var parser = Harmony.StreamableParser()
        for t in tokens {
            try parser.process(t)
        }
        try parser.processEOS()
        let msgs = parser.messages
        try #require(msgs.count == 1)
        #expect(msgs[0].author.role == .assistant)
        #expect(msgs[0].channel == "commentary")
        #expect(msgs[0].recipient == "functions.get_weather")
        #expect(msgs[0].contentType == "<|constrain|>json")
        #expect(msgs[0].content == [.text("{\"latitude\":48.8566,\"longitude\":2.3522}")])
    }

    @Test
    func toolCallWithChannelBeforeRecipientAndConstrainAdjacent() async throws {
        // harmony_tests.rs::test_tool_call_with_channel_before_recipient_and_constrain_adjacent
        let text =
            "<|start|>assistant to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{\"location\": \"Tokyo\"}<|end|>"
        let tokens = tokenizeWithSpecials(text)
        var parser = Harmony.StreamableParser()
        for t in tokens {
            try parser.process(t)
        }
        try parser.processEOS()
        let msgs = parser.messages
        try #require(msgs.count == 1)
        #expect(msgs[0].author.role == .assistant)
        #expect(msgs[0].channel == "commentary")
        #expect(msgs[0].recipient == "functions.get_weather")
        #expect(msgs[0].contentType == "<|constrain|>json")
        #expect(msgs[0].content == [.text("{\"location\": \"Tokyo\"}")])
    }

    @Test
    func toolResponseParsing() async throws {
        // harmony_tests.rs::test_tool_response_parsing
        let text =
            "<|start|>browser.search to=assistant<|channel|>commentary<|message|>{\"result\": \"https://openai.com/\"}<|end|>"
        let tokens = tokenizeWithSpecials(text)
        var parser = Harmony.StreamableParser()
        for t in tokens {
            try parser.process(t)
        }
        try parser.processEOS()

        let msgs = parser.messages
        try #require(msgs.count == 1)
        #expect(msgs[0].author.role == .tool)
        #expect(msgs[0].author.name == "browser.search")
        #expect(msgs[0].channel == "commentary")
        #expect(msgs[0].recipient == "assistant")
        #expect(msgs[0].content == [.text("{\"result\": \"https://openai.com/\"}")])
    }

    @Test
    func startingRoleOverrideWithoutRoleToken() async throws {
        let text = "<|start|><|message|>Hello!<|end|>"
        let tokens = tokenizeWithSpecials(text)
        var parser = Harmony.StreamableParser(startingRole: .assistant)
        for t in tokens {
            try parser.process(t)
        }
        try parser.processEOS()
        let msgs = parser.messages
        try #require(msgs.count == 1)
        #expect(msgs[0].author.role == .assistant)
        #expect(msgs[0].content == [.text("Hello!")])
    }

    @Test
    func handlesSimplifiedMLXStream() async throws {
        let rawTokens = [
            "<|channel|>", "analysis", "<|message|>", "Hello", "<|end|>",
            "<|start|>", "assistant", "<|channel|>", "final", "<|message|>",
            "Goodbye",
        ]

        var parser = Harmony.StreamableParser(startingRole: .assistant)
        for t in rawTokens {
            try parser.process(t)
        }
        try parser.processEOS()

        let msgs = parser.messages
        try #require(msgs.count == 2)

        #expect(msgs[0].author.role == .assistant)
        #expect(msgs[0].channel == "analysis")
        #expect(msgs[0].content == [.text("Hello")])

        #expect(msgs[1].author.role == .assistant)
        #expect(msgs[1].channel == "final")
        #expect(msgs[1].content == [.text("Goodbye")])
    }

    @Test
    func handlesMLXStream() async throws {
        var parser = Harmony.StreamableParser(startingRole: .assistant)
        for t in mlxStreamingResponse {
            try parser.process(t)
        }
        try parser.processEOS()

        let msgs = parser.messages
        try #require(msgs.count == 2)

        #expect(msgs[0].author.role == .assistant)
        #expect(msgs[0].channel == "analysis")
        #expect(msgs[1].author.role == .assistant)
        #expect(msgs[1].channel == "final")
    }

    @Test
    func handlesMLXSimpleStreamingResponseWithTool() async throws {
        let responses = mlxSimpleStreamingResponseWithTool.responses(
            startingRole: .assistant
        )

        var reasoning = ""
        var text = ""
        var toolCall: ToolCall?
        for response in responses {
            switch response {
            case let .reasoning(r): reasoning += r
            case let .text(t): text += t
            case let .toolCall(t): toolCall = t
            }
        }

        #expect(!reasoning.isEmpty)
        #expect(text.isEmpty)

        let expectedToolCall = ToolCall(function: ToolCall.Function(
            name: "get_current_weather",
            arguments: ["location": "Paris, France"]
        ))
        #expect(toolCall == expectedToolCall)
    }

    @Test
    func handlesMLXStreamWithToolFixture() async throws {
        let responses = mlxStreamingResponseWithTool.responses(
            startingRole: .assistant
        )

        var reasoning = ""
        var text = ""
        var toolCall: ToolCall?
        for response in responses {
            switch response {
            case let .reasoning(r): reasoning += r
            case let .text(t): text += t
            case let .toolCall(t): toolCall = t
            }
        }

        #expect(!reasoning.isEmpty)
        #expect(text.isEmpty)

        let expectedToolCall = ToolCall(function: ToolCall.Function(
            name: "get_current_weather",
            arguments: ["location": "Paris, France"]
        ))
        #expect(toolCall == expectedToolCall)
    }

    @Test
    func streamParsedTokensToFakeClient() async throws {
        let rawTokens = [
            "<|channel|>", "analysis", "<|message|>", "Hello", "<|end|>",
            "<|start|>", "assistant", "<|channel|>", "final", "<|message|>",
            "Goodbye",
        ]

        let (stream, client) = AsyncStream.makeStream(of: Response.self)

        let task = Task {
            var reasoning = ""
            var text = ""
            for try await response in stream {
                switch response {
                case let .reasoning(r): reasoning += r
                case let .text(t): text += t
                case .toolCall: #expect(Bool(false))
                }
            }
            #expect(reasoning == "Hello")
            #expect(text == "Goodbye")
        }

        let responses = rawTokens.responses(startingRole: .assistant)
        for response in responses {
            client.yield(response)
        }
        client.finish()

        try await task.value
    }
}

private extension Sequence where Element: StringProtocol {
    func responses(
        startingRole: Harmony.Role? = nil
    ) -> AnySequence<Response> {
        AnySequence {
            var i = makeIterator()
            var parser = Harmony.StreamableParser(
                startingRole: startingRole
            )
            return AnyIterator<Response> {
                // NOTE: LLMs are happy to go on and on even after responding
                //       with a tool call. So, we need to check if the last
                //       token was a tool call token. If it was a tool call
                //       token, we need to stop inference.
                if let last = parser.tokens.last,
                   case let .special(s) = Harmony.Token(last),
                   s.isToolCall
                {
                    return nil
                }

                while let token = i.next() {
                    do {
                        let messageCount = parser.messages.count
                        try parser.process(String(token))

                        if let delta = parser.delta {
                            if parser.channel == "analysis" {
                                return .reasoning(delta)
                            } else if parser.channel == "final" {
                                return .text(delta)
                            }
                        }

                        guard parser.messages.count > messageCount,
                              let lastMessage = parser.messages.last,
                              lastMessage.author.role == .assistant,
                              let recipient = lastMessage.recipient,
                              recipient.hasPrefix("functions.")
                        else { continue }

                        let functionName = String(recipient.dropFirst("functions.".count))
                        let decoder = JSONDecoder()
                        if case let .text(content) = lastMessage.content.first,
                           let jsonData = content.data(using: .utf8),
                           let jsonObject = try? decoder.decode(JSONValue.self, from: jsonData),
                           let args = jsonObject.anyValue as? [String: Any]
                        {
                            let toolCall = ToolCall(
                                function: ToolCall.Function(
                                    name: functionName,
                                    arguments: args
                                )
                            )
                            return .toolCall(toolCall)
                        }

                        // NOTE: Always break on any tool call, even if it
                        //       is formatted incorrectly.
                        break
                    } catch {
                        #expect(Bool(false), "unexpected error: \(String(describing: error))")
                        try? parser.processEOS()
                        return nil
                    }
                }
                try? parser.processEOS()
                return nil
            }
        }
    }
}

private extension HarmonyTests {
    private func tokenizeWithSpecials(_ text: String) -> [String] {
        var tokens: [String] = []
        var i = text.startIndex
        func pushText(_ s: String) { if !s.isEmpty { tokens.append(s) } }
        while i < text.endIndex {
            if text[i] == "<" {
                // possible special
                if let close = text[i...].firstIndex(of: ">") {
                    let candidate = String(text[i ... close])
                    if candidate.hasPrefix("<|"), candidate.hasSuffix("|>") {
                        tokens.append(candidate)
                        i = text.index(after: close)
                        continue
                    }
                }
            }
            // collect plain run until next '<'
            let next = text[i...].firstIndex(of: "<") ?? text.endIndex
            pushText(String(text[i ..< next]))
            i = next
        }
        return tokens
    }

    private var mlxStreamingResponse: [String] {
        [
            "<|channel|>", "analysis", "<|message|>", "We", " need", " to", " answer", " \"",
            "What", " is", " the", " meaning", " of", " life", "?\"", " We", " should",
            " respond", " in", " a", " helpful", " manner", ".", " Possibly", " philosophical",
            ".", " Provide", " perspective", ".", " Use", " empathy", ".", "<|end|>",
            "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "The", " question",
            " “", "What", " is", " the", " meaning", " of", " life", "?”", " has",
            " fascinated", " philosophers", ",", " scientists", ",", " artists", ",", " and",
            " ordinary", " people", " for", " mill", "ennia", ".", " There", " isn", "’t", " a",
            " single", ",", " universally", " accepted", " answer", ",", " but", " there",
            " are", " many", " ways", " to", " think", " about", " it", " that", " can",
            " help", " you", " find", " your", " own", " sense", " of", " purpose", " and",
            " fulfillment", ".\n\n", "|", " Perspective", " |", " Key", " Ideas", " |", " How",
            " It", " Can", " Help", " You", " |\n", "|", "-------------", "|", "-----------",
            "|", "----------------", "-----", "|\n", "|", " **", "Exist", "ential", "ist", "**",
            " |", " Life", " has", " no", " inherent", " meaning", ";", " we", " create", " it",
            " through", " choices", ",", " authenticity", ",", " and", " responsibility", ".",
            " |", " Encour", "ages", " you", " to", " take", " ownership", " of", " your",
            " path", " and", " make", " decisions", " that", " align", " with", " your",
            " values", ".", " |\n", "|", " **", "Human", "ist", "**", " |", " Meaning",
            " comes", " from", " human", " connections", ",", " creativity", ",", " and",
            " the", " pursuit", " of", " knowledge", ".", " |", " Insp", "ires", " you", " to",
            " nurture", " relationships", ",", " learn", ",", " and", " contribute", " to",
            " the", " well", "‑", "being", " of", " others", ".", " |\n", "|", " **", "Rel",
            "igious", " &", " Spiritual", "**", " |", " Meaning", " is", " often", " tied",
            " to", " a", " higher", " power", ",", " cosmic", " order", ",", " or",
            " spiritual", " growth", " (", "e", ".g", ".,", " fulfilling", " God", "’s",
            " will", ",", " achieving", " enlightenment", ").", " |", " Provides", " a",
            " framework", " for", " moral", " guidance", ",", " community", ",", " and",
            " rituals", " that", " can", " bring", " comfort", " and", " direction", ".",
            " |\n", "|", " **", "Bi", "ological", "**", " |", " From", " an", " evolutionary",
            " standpoint", ",", " the", " “", "purpose", "”", " of", " life", " is", " to",
            " survive", ",", " reproduce", ",", " and", " propagate", " genes", ".", " |",
            " Helps", " you", " appreciate", " the", " instinct", "ual", " drives", " that",
            " shape", " behavior", " and", " can", " inform", " how", " you", " balance",
            " personal", " goals", " with", " natural", " rhythms", ".", " |\n", "|", " **",
            "Narr", "ative", "**", " |", " We", " give", " life", " meaning", " by",
            " weaving", " stories", " around", " our", " experiences", "—", "stories", " that",
            " give", " context", ",", " purpose", ",", " and", " continuity", ".", " |",
            " Encour", "ages", " you", " to", " reflect", " on", " your", " own", " story",
            ",", " set", " goals", ",", " and", " write", " a", " narrative", " that", " feels",
            " coherent", " and", " satisfying", ".", " |\n", "|", " **", "Ther", "apeut", "ic",
            " (", "e", ".g", ".,", " Positive", " Psychology", ")**", " |", " Meaning",
            " arises", " from", " engagement", " (", "flow", "),", " relationships", ",",
            " and", " a", " sense", " of", " accomplishment", ".", " |", " Offers", " concrete",
            " practices", "—", "grat", "itude", " journ", "aling", ",", " goal", " setting",
            ",", " skill", " development", "—to", " enhance", " well", "‑", "being", ".",
            " |\n\n", "###", " Tips", " for", " Craft", "ing", " Your", " Own", " Meaning",
            "\n\n", "1", ".", " **", "Reflect", " on", " Core", " Values", "**", "  \n", "  ",
            " What", " principles", " matter", " most", " to", " you", "?", " Integrity", ",",
            " creativity", ",", " compassion", ",", " adventure", "?", " Align", "ing",
            " daily", " actions", " with", " these", " values", " gives", " a", " sense",
            " of", " purpose", ".\n\n", "2", ".", " **", "Set", " Meaning", "ful", " Goals",
            "**", "  \n", "  ", " Choose", " goals", " that", " stretch", " you", ",",
            " challenge", " you", ",", " and", " resonate", " with", " your", " values", ".",
            " Break", " them", " into", " actionable", " steps", " and", " celebrate",
            " progress", ".\n\n", "3", ".", " **", "Cult", "ivate", " Relationships", "**",
            "  \n", "  ", " Deep", ",", " authentic", " connections", " provide", " emotional",
            " support", ",", " shared", " joy", ",", " and", " a", " sense", " of",
            " belonging", "—", "key", " ingredients", " to", " a", " meaningful", " life",
            ".\n\n", "4", ".", " **", "Find", " Flow", "**", "  \n", "  ", " Engage", " in",
            " activities", " where", " you", " lose", " track", " of", " time", " because",
            " you", "’re", " fully", " absorbed", ".", " This", " state", " of", " flow",
            " is", " often", " linked", " to", " high", " satisfaction", " and", " purpose",
            ".\n\n", "5", ".", " **", "Give", " Back", "**", "  \n", "  ", " Vol", "unte",
            "ering", ",", " mentoring", ",", " or", " simply", " helping", " others", " can",
            " create", " a", " sense", " of", " impact", " and", " connection", " that",
            " enrich", "es", " your", " life", ".\n\n", "6", ".", " **", "Practice", " Mind",
            "fulness", "**", "  \n", "  ", " Being", " present", " helps", " you", " notice",
            " the", " small", " moments", " that", " accumulate", " into", " a", " rich", ",",
            " meaningful", " tapestry", ".\n\n", "7", ".", " **", "Re", "visit", " and",
            " Rev", "ise", "**", "  \n", "  ", " Your", " sense", " of", " meaning", " can",
            " evolve", ".", " Period", "ically", " reass", "ess", " your", " goals", ",",
            " values", ",", " and", " relationships", " to", " stay", " aligned", " with",
            " your", " current", " self", ".\n\n", "###", " A", " Thought", " Experiment",
            "\n\n", "Imagine", " you", "’re", " an", " author", " writing", " a", " book",
            " about", " your", " life", ".", " The", " first", " chapter", " is", " your",
            " childhood", ";", " the", " next", ",", " your", " teenage", " years", ";",
            " then", " adulthood", ".", " What", " themes", " would", " you", " highlight",
            "?", " Love", "?", " Loss", "?", " Triumph", "?", " By", " framing", " your",
            " life", " as", " a", " narrative", ",", " you", " can", " identify", " recurring",
            " motifs", " and", " decide", " what", " you", " want", " the", " story", " to",
            " ultimately", " convey", ".\n\n", "###", " Bottom", " Line", "\n\n", "The",
            " meaning", " of", " life", " is", " less", " a", " single", " destination",
            " and", " more", " a", " journey", " of", " intentional", " living", ".", " It",
            "’s", " about", " consciously", " choosing", " how", " you", " spend", " your",
            " time", ",", " whom", " you", " connect", " with", ",", " and", " what", " you",
            " strive", " toward", ".", " By", " aligning", " your", " actions", " with",
            " what", " truly", " matters", " to", " you", ",", " you", " can", " create",
            " a", " life", " that", " feels", " purposeful", " and", " deeply",
            " satisfying", ".",
        ]
    }

    private var mlxSimpleStreamingResponseWithTool: [String] {
        [
            "<|channel|>", "analysis", "<|message|>", "We", " need", " to", " use", " get",
            "_current", "_weather", " tool", ".", " The", " location", " is", " \"", "Paris",
            ",", " France", "\".", " The", " user", " didn\'t", " specify", " unit", ".",
            " We", " can", " default", " to", " c", "elsius", " or", " ask", "?", " The",
            " instruction", " says", " we", " must", " fetch", " weather", " data", ".",
            " So", " we", " should", " call", " tool", ".", "<|end|>", "<|start|>", "assistant",
            "<|channel|>", "comment", "ary", " to", "=", "functions", ".get", "_current",
            "_weather", " ", "<|constrain|>", "json", "<|message|>", "{\"", "location", "\":\"",
            "Paris", ",", " France", "\"}", "<|call|>", "comment", "ary", "<|message|>", "We",
            " need", " to", " output", " the", " result", ".", "<|end|>", "<|start|>",
            "assistant", "<|channel|>", "final", "<|message|>", "Here", "’s", " the",
            " current", " weather", " in", " Paris", ",", " France", ":\n\n", "-", " **",
            "Temperature", ":**", " ", "18", " ", "°C", " (", "64", " ", "°F", ")", "  \n",
            "-", " **", "Conditions", ":**", " Part", "ly", " cloudy", "  \n", "-", " **",
            "Humidity", ":**", " ", "55", "%", "  \n", "-", " **", "Wind", ":**", " ", "12",
            " ", "km", "/h", " (", "7", " ", "mph", ")", " from", " the", " northwest",
            "  \n\n", "Let", " me", " know", " if", " you", "’d", " like", " a", " forecast",
            " for", " the", " next", " few", " days", " or", " any", " other", " details", "!",
        ]
    }

    private var mlxStreamingResponseWithTool: [String] {
        [
            "<|channel|>", "analysis", "<|message|>", "User", ":", " \"", "What", " is", " the",
            " weather", " in", " Paris", ",", " France", "?\"", " We", " need", " to", " use",
            " get", "_current", "_weather", " tool", ".", " The", " location", " should", " be",
            " \"", "Paris", ",", " France", "\"", " or", " \"", "Paris", ",", " France", "\"",
            " but", " spec", " says", " city", " and", " state", " e", ".g", ".", " San",
            " Francisco", ",", " CA", ".", " But", " we", " have", " a", " city", " and",
            " country", ".", " The", " tool", " expects", " location", " string", ".",
            " We\'ll", " pass", " \"", "Paris", ",", " France", "\".", " And", " unit", "?",
            " We", " can", " default", ".", " We\'ll", " call", " get", "_current",
            "_weather", ".", "<|end|>", "<|start|>", "assistant", "<|channel|>",
            "comment", "ary", " to", "=", "functions", ".get", "_current", "_weather", " ",
            "<|constrain|>", "json", "<|message|>", "{\"", "location", "\":\"", "Paris", ",",
            " France", "\"}", "<|call|>", "comment", "ary", "<|channel|>", "comment", "ary",
            " to", "=", "functions", ".get", "_current", "_weather", " ", "<|constrain|>",
            "json", "<|message|>", "{}", "<|call|>", "comment", "ary", "<|channel|>",
            "analysis", "<|message|>", "The", " tool", " returned", " empty", ".", " Maybe",
            " the", " tool", " didn\'t", " return", " data", "?", " Possibly", " the", " tool",
            " fails", ".", " We", " can", " respond", " with", " placeholder", " or", " ask",
            " to", " try", " again", ".", " But", " maybe", " we", " need", " to", " provide",
            " default", " unit", " as", " \"", "c", "elsius", "\".", " Let\'s", " try",
            " again", ".", "<|end|>", "<|start|>", "assistant", "<|channel|>", "comment",
            "ary", " to", "=", "functions", ".get", "_current", "_weather", " ",
            "<|constrain|>", "json", "<|message|>", "{\"", "location", "\":\"", "Paris", ",",
            " France", "\",\"", "unit", "\":\"", "c", "elsius", "\"}", "<|call|>", "comment",
            "ary", "<|channel|>", "comment", "ary", " to", "=", "functions", ".get", "_current",
            "_weather", " ", "<|constrain|>", "json", "<|message|>", "{}", "<|call|>",
            "comment", "ary", "<|channel|>", "analysis", "<|message|>", "No", " data",
            " returned", ".", " Possibly", " the", " tool", " is", " not", " implemented", ".",
            " We\'ll", " respond", " that", " we", " cannot", " fetch", ".", " But",
            " instructions", ":", " must", " use", " get", "_current", "_weather", " tool",
            ".", " We", " already", " used", ".", " Now", " respond", " with", " the",
            " result", ".", " Since", " no", " data", ",", " we", " can", " say", " \"", "I",
            " couldn\'t", " retrieve", " the", " data", ".\"", " But", " we", " must",
            " not", " mention", " tool", " usage", ".", " We\'ll", " say", ":", " \"", "I\'m",
            " sorry", ",", " I", " couldn\'t", " fetch", " the", " current", " weather",
            " for", " Paris", ",", " France", " at", " this", " time", ".\"", "<|end|>",
            "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "I", "’m",
            " sorry", ",", " but", " I", " couldn", "’t", " retrieve", " the", " current",
            " weather", " information", " for", " Paris", ",", " France", " right", " now", ".",
        ]
    }
}
