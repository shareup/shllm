import Foundation
import MLXLMCommon

public enum Python {
    /// Consumes and returns the first complete Python-style function call at the
    /// start of `buffer`, mutating `buffer` to remove the consumed text
    /// (including surrounding list brackets, optional commas, and whitespace).
    ///
    /// If no complete call is available yet, returns `nil` and leaves `buffer`
    /// (aside from trimming leading brackets/commas/whitespace) as-is.
    public static func parseFunctionCall(_ buffer: inout String) -> ToolCall? {
        let copy = buffer

        var i = copy.startIndex
        skipWhitespace(copy, &i)

        while i < copy.endIndex {
            let ch = copy[i]
            if ch == "[" || ch == "]" || ch == "," || ch.isWhitespace {
                i = copy.index(after: i)
                continue
            }
            break
        }

        if i >= copy.endIndex {
            buffer = ""
            return nil
        }

        var j = i
        guard let nameRange = readIdentifier(copy, &j) else {
            buffer = String(copy[i...])
            return nil
        }
        skipWhitespace(copy, &j)
        guard j < copy.endIndex, copy[j] == "(" else {
            buffer = String(copy[i...])
            return nil
        }
        let open = j
        guard let close = findMatchingParen(in: copy, openIndex: open) else {
            buffer = String(copy[i...])
            return nil
        }

        let callStr = String(copy[nameRange.lowerBound ..< copy.index(after: close)])
        guard let call = parseSingleCall(callStr) else {
            buffer = String(copy[i...])
            return nil
        }

        var k = copy.index(after: close)
        skipWhitespace(copy, &k)
        if k < copy.endIndex, copy[k] == "," {
            k = copy.index(after: k)
            skipWhitespace(copy, &k)
        }

        while k < copy.endIndex,
              copy[k] == "]" || copy[k].isWhitespace { k = copy.index(after: k) }

        buffer = String(copy[k...])
        return call
    }
}

private extension Python {
    static func stripListBrackets(_ s: String) -> String {
        guard s.first == "[", s.last == "]" else { return s }
        return String(s.dropFirst().dropLast())
    }

    static func parseSingleCall(_ s: String) -> ToolCall? {
        let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let open = trimmed.firstIndex(of: "(") else { return nil }
        let name = trimmed[..<open].trimmingCharacters(in: .whitespacesAndNewlines)
        guard let close = findMatchingParen(in: trimmed, openIndex: open) else { return nil }
        let argsBody = String(trimmed[trimmed.index(after: open) ..< close])
        let args = parseArgs(argsBody)
        return ToolCall(function: ToolCall.Function(name: String(name), arguments: args))
    }

    static func parseArgs(_ body: String) -> [String: Any] {
        let parts = splitTopLevel(body, on: ",")
        var dict: [String: Any] = [:]
        for part in parts {
            let trimmed = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            guard let eq = indexOfTopLevelEquals(in: trimmed) else { continue }
            let key = trimmed[..<eq].trimmingCharacters(in: .whitespacesAndNewlines)
            let valueRaw = trimmed[trimmed.index(after: eq)...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            dict[String(key)] = parseValue(valueRaw)
        }
        return dict
    }

    static func parseValue(_ s: String) -> Any {
        guard let first = s.first else { return "" }
        if first == "\"" || first == "'" { return unquoteString(s) }

        let lower = s.lowercased()
        if lower == "true" { return true }
        if lower == "false" { return false }
        if lower == "none" || lower == "null" { return NSNull() }
        if let intVal = Int(s) { return intVal }
        if let dblVal = Double(s) { return dblVal }
        return s
    }

    static func unquoteString(_ s: String) -> String {
        guard let q = s.first, q == "\"" || q == "'", s.last == q else { return s }
        let inner = s.dropFirst().dropLast()
        var result = ""
        result.reserveCapacity(inner.count)
        var escape = false
        for ch in inner {
            if escape {
                switch ch {
                case "n": result.append("\n")
                case "t": result.append("\t")
                case "r": result.append("\r")
                case "\\": result.append("\\")
                case "\"": result.append("\"")
                case "'": result.append("'")
                default: result.append(ch)
                }
                escape = false
            } else if ch == "\\" {
                escape = true
            } else {
                result.append(ch)
            }
        }
        return result
    }

    static func skipWhitespace(_ s: String, _ i: inout String.Index) {
        while i < s.endIndex, s[i].isWhitespace {
            i = s.index(after: i)
        }
    }

    static func readIdentifier(_ s: String, _ i: inout String.Index) -> Range<String.Index>? {
        guard i < s.endIndex else { return nil }
        let start = i
        // identifiers: [A-Za-z_][A-Za-z0-9_]*
        func isStart(_ c: Character) -> Bool { c.isLetter || c == "_" }
        func isCont(_ c: Character) -> Bool { c.isLetter || c.isNumber || c == "_" }
        guard isStart(s[i]) else { return nil }
        i = s.index(after: i)
        while i < s.endIndex, isCont(s[i]) {
            i = s.index(after: i)
        }
        return start ..< i
    }

    static func splitTopLevel(_ s: String, on sep: Character) -> [String] {
        var out: [String] = []
        var start = s.startIndex
        var i = s.startIndex
        var depthParen = 0, depthBrace = 0, depthBracket = 0
        var quote: Character?
        var escape = false

        func flush(_ end: String.Index) {
            let piece = s[start ..< end]
            out.append(String(piece).trimmingCharacters(in: .whitespacesAndNewlines))
            start = s.index(after: end)
        }

        while i < s.endIndex {
            let ch = s[i]
            if let q = quote {
                if escape { escape = false }
                else if ch == "\\" { escape = true }
                else if ch == q { quote = nil }
            } else {
                switch ch {
                case "'", "\"": quote = ch
                case "(": depthParen += 1
                case ")": if depthParen > 0 { depthParen -= 1 }
                case "[": depthBracket += 1
                case "]": if depthBracket > 0 { depthBracket -= 1 }
                case "{": depthBrace += 1
                case "}": if depthBrace > 0 { depthBrace -= 1 }
                default:
                    if ch == sep, depthParen == 0, depthBrace == 0, depthBracket == 0 {
                        flush(i)
                    }
                }
            }
            i = s.index(after: i)
        }
        if start <= s.endIndex {
            out.append(String(s[start...]).trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return out.filter { !$0.isEmpty }
    }

    static func findMatchingParen(
        in s: String,
        openIndex: String.Index
    ) -> String.Index? {
        var i = s.index(after: openIndex)
        var depth = 1
        var quote: Character?
        var escape = false
        while i < s.endIndex {
            let ch = s[i]
            if let q = quote {
                if escape { escape = false }
                else if ch == "\\" { escape = true }
                else if ch == q { quote = nil }
            } else {
                switch ch {
                case "'", "\"": quote = ch
                case "(": depth += 1
                case ")": depth -= 1; if depth == 0 { return i }
                default: break
                }
            }
            i = s.index(after: i)
        }
        return nil
    }

    static func indexOfTopLevelEquals(in s: String) -> String.Index? {
        var i = s.startIndex
        var depthParen = 0, depthBrace = 0, depthBracket = 0
        var quote: Character?
        var escape = false
        while i < s.endIndex {
            let ch = s[i]
            if let q = quote {
                if escape { escape = false }
                else if ch == "\\" { escape = true }
                else if ch == q { quote = nil }
            } else {
                switch ch {
                case "'", "\"": quote = ch
                case "(": depthParen += 1
                case ")": if depthParen > 0 { depthParen -= 1 }
                case "[": depthBracket += 1
                case "]": if depthBracket > 0 { depthBracket -= 1 }
                case "{": depthBrace += 1
                case "}": if depthBrace > 0 { depthBrace -= 1 }
                case "=": if depthParen == 0, depthBrace == 0, depthBracket == 0 { return i }
                default: break
                }
            }
            i = s.index(after: i)
        }
        return nil
    }
}
