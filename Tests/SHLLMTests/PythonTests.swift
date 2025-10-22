import Foundation
import MLXLLM
import MLXLMCommon
@testable import SHLLM
import Testing

@Suite
struct PythonTests {
    @Test
    func parseSingleCallWithDoubleQuotes() throws {
        var s = #"[get_current_weather(location="Paris, France")]"#
        let call = try #require(Python.parseFunctionCall(&s))
        #expect(call.function.name == "get_current_weather")
        #expect(call.function.arguments["location"] == .string("Paris, France"))
        #expect(s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test
    func parseSingleCallWithSingleQuotesAndEscapes() throws {
        var s = #"[find(name='O\'Hara, "Alex"')]"#
        let call = try #require(Python.parseFunctionCall(&s))
        #expect(call.function.name == "find")
        #expect(call.function.arguments["name"] == .string("O'Hara, \"Alex\""))
        #expect(s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test
    func parseMultipleCallsWithVariedWhitespace() throws {
        var s = #"[ foo(a="1") ,bar(b='x, y'),  baz(c="z") ]"#
        var out: [ToolCall] = []
        while let call = Python.parseFunctionCall(&s) {
            out.append(call)
        }
        #expect(out.count == 3)
        #expect(out[0].function.name == "foo")
        #expect(out[1].function.name == "bar")
        #expect(out[2].function.name == "baz")
        #expect(out[1].function.arguments["b"] == .string("x, y"))
        #expect(s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test
    func parseSingleCallWithoutBrackets() throws {
        var s = #"get_latest_news(query="Apple", sortBy="popularity")"#
        let call = try #require(Python.parseFunctionCall(&s))
        #expect(call.function.name == "get_latest_news")
        #expect(call.function.arguments["query"] == .string("Apple"))
        #expect(call.function.arguments["sortBy"] == .string("popularity"))
        #expect(s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test
    func parseNumbersBooleansAndNullValues() throws {
        var s = #"[fn(count=3, pi=3.14, ok=true, off=false, none=null)]"#
        let call = try #require(Python.parseFunctionCall(&s))
        #expect((call.function.arguments["count"]?.anyValue as? Int) == 3)
        #expect((call.function.arguments["pi"]?.anyValue as? Double) == 3.14)
        #expect((call.function.arguments["ok"]?.anyValue as? Bool) == true)
        #expect((call.function.arguments["off"]?.anyValue as? Bool) == false)
        #expect(call.function.arguments["none"] == .null)
        #expect(s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test
    func parseSingleCallWithTrailingCommaAndExtraSpaces() throws {
        var s = #"[get_stock_price(symbol="AAPL"), ]"#
        let call = try #require(Python.parseFunctionCall(&s))
        #expect(call.function.name == "get_stock_price")
        #expect(call.function.arguments["symbol"] == .string("AAPL"))
        #expect(s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test
    func parseTwoCallsIncrementally() throws {
        var buf = #"[foo(a="1"), bar(b="2")]"#
        let c1 = Python.parseFunctionCall(&buf)
        let call1 = try #require(c1)
        #expect(call1.function.name == "foo")
        #expect(call1.function.arguments["a"] == .string("1"))

        // Buffer should now contain only the remaining call
        #expect(buf.contains("bar(b=\"2\")"))

        let c2 = Python.parseFunctionCall(&buf)
        let call2 = try #require(c2)
        #expect(call2.function.name == "bar")
        #expect(call2.function.arguments["b"] == .string("2"))

        // Fully consumed
        #expect(buf.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test
    func parseFirstCallWhenStringHasCommas() throws {
        var buf = #"[foo(msg='a, b, c'), bar(n=1)]"#
        let c1 = Python.parseFunctionCall(&buf)
        let call1 = try #require(c1)
        #expect(call1.function.name == "foo")
        #expect(call1.function.arguments["msg"] == .string("a, b, c"))
        #expect(buf.hasPrefix("bar(n=1)"))
    }

    @Test
    func parseFirstCallSkippingNoiseAndBrackets() throws {
        var buf = "  [  ,  foo(a=1)  ,  ]  "
        let c1 = Python.parseFunctionCall(&buf)
        let call1 = try #require(c1)
        #expect(call1.function.name == "foo")
        #expect((call1.function.arguments["a"]?.anyValue as? Int) == 1)
        #expect(buf.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test
    func parseFirstCallWhenItBecomesComplete() throws {
        var buf = "[foo(a=\"1\"" // missing closing paren
        let c1 = Python.parseFunctionCall(&buf)
        #expect(c1 == nil)
        // Should retain the content (minus leading bracket/space) for later completion
        #expect(buf.contains("foo(a=\"1\""))

        buf += ") , bar(b=2)]"
        let c2 = Python.parseFunctionCall(&buf)
        let call = try #require(c2)
        #expect(call.function.name == "foo")
        #expect(call.function.arguments["a"] == .string("1"))
        // After consuming first, remainder should include second call only
        #expect(buf.hasPrefix("bar(b=2)"))
    }

    @Test
    func parseFirstCallWithNestedParensInString() throws {
        var buf = #"[fn(expr="(a, b), c"), fn2(x=3)]"#
        let c1 = Python.parseFunctionCall(&buf)
        let call1 = try #require(c1)
        #expect(call1.function.name == "fn")
        #expect(call1.function.arguments["expr"] == .string("(a, b), c"))
        #expect(buf.hasPrefix("fn2(x=3)"))
    }
}
