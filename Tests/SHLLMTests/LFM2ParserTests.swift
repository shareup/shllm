import Foundation
import MLXLLM
import MLXLMCommon
@testable import SHLLM
import Testing

@Suite
struct LFM2ParserTests {
    private func parseAll(
        _ parser: LLM<LFM2MoEModel>.ResponseParser,
        gens: [Generation]
    ) -> [Response] {
        gens.compactMap { parser.parse($0) }
    }

    @Test
    func singleCall_emitOnce_thenText() throws {
        let p = LLM<LFM2MoEModel>.lfm2Parser
        let gens: [Generation] = [
            .chunk("<|tool_call_start|>"),
            .chunk("[get_candidate_status(candidate_id=\"123\")]"),
            .chunk("<|tool_call_end|>"),
            .chunk("Done."),
        ]

        let out = parseAll(p, gens: gens)
        #expect(out.count == 2)

        guard case let .toolCall(call) = out.first else { return #expect(Bool(false)) }
        #expect(call.function.name == "get_candidate_status")
        #expect(call.function.arguments["candidate_id"] == .string("123"))

        guard case let .text(t) = out.last else { return #expect(Bool(false)) }
        #expect(t == "Done.")
    }

    @Test
    func multipleCalls_emitted_incrementally() throws {
        let p = LLM<LFM2MoEModel>.lfm2Parser
        var results: [Response] = []

        // Start block
        if let r = p.parse(.chunk("<|tool_call_start|>")) { results.append(r) }
        #expect(results.isEmpty)

        // First chunk contains a complete first call and a trailing comma.
        if let r = p.parse(.chunk("[foo(a=\"1\"), ")) { results.append(r) }
        guard case let .toolCall(first) = results.last else { return #expect(Bool(false)) }
        #expect(first.function.name == "foo")
        #expect(first.function.arguments["a"] == .string("1"))

        // Second chunk completes the second call.
        if let r = p.parse(.chunk("bar(b=\"2\")]")) { results.append(r) }
        guard case let .toolCall(second) = results.last else { return #expect(Bool(false)) }
        #expect(second.function.name == "bar")
        #expect(second.function.arguments["b"] == .string("2"))

        // End block should not emit anything further.
        #expect(p.parse(.chunk("<|tool_call_end|>")) == nil)
    }

    @Test
    func partialAcrossChunks_emitsWhenComplete() throws {
        let p = LLM<LFM2MoEModel>.lfm2Parser
        var emitted: [Response] = []

        _ = p.parse(.chunk("<|tool_call_start|>"))
        // Incomplete first call
        #expect(p.parse(.chunk("[foo(a=\"")) == nil)

        // Completing first call and starting second
        if let r = p.parse(.chunk("1\") , bar(b=2")) { emitted.append(r) }
        guard case let .toolCall(first) = emitted.last else { return #expect(Bool(false)) }
        #expect(first.function.name == "foo")
        #expect(first.function.arguments["a"] == .string("1"))

        // Finish second call
        if let r = p.parse(.chunk(")]")) { emitted.append(r) }
        guard case let .toolCall(second) = emitted.last else { return #expect(Bool(false)) }
        #expect(second.function.name == "bar")
        #expect((second.function.arguments["b"]?.anyValue as? Int) == 2)

        // End token
        #expect(p.parse(.chunk("<|tool_call_end|>")) == nil)
    }
}
