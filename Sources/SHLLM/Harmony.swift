import Foundation

enum Harmony {
    enum Role: String, Codable, Equatable, Sendable {
        case system
        case developer
        case user
        case assistant
        case tool
    }

    struct Author: Codable, Equatable, Sendable {
        var role: Role
        var name: String?

        init(role: Role, name: String? = nil) {
            self.role = role
            self.name = name
        }
    }

    enum Content: Codable, Equatable, Sendable {
        case text(String)
    }

    struct Message: Codable, Equatable, Sendable {
        var author: Author
        var recipient: String?
        var channel: String?
        var contentType: String?
        var content: [Content]

        init(
            author: Author,
            recipient: String? = nil,
            channel: String? = nil,
            contentType: String? = nil,
            content: [Content]
        ) {
            self.author = author
            self.recipient = recipient
            self.channel = channel
            self.contentType = contentType
            self.content = content
        }
    }
}

extension Harmony {
    struct StreamableParser: Equatable, Sendable {
        struct ParsedHeader: Equatable, Sendable {
            var author: Harmony.Author
            var recipient: String?
            var channel: String?
            var contentType: String?
        }

        enum StreamState: Equatable, Sendable {
            case expectStart
            case header(headerTokens: [Harmony.Token])
            case content(header: ParsedHeader, content: String)
        }

        private(set) var messages: [Harmony.Message] = []
        private(set) var tokens: [String] = []
        private(set) var delta: String?

        var content: String {
            if case let .content(_, content) = state {
                return content
            }
            return ""
        }

        var role: Harmony.Role? {
            switch state {
            case let .content(header, _): header.author.role
            default: nextRole
            }
        }

        var contentType: String? {
            if case let .content(header, _) = state {
                return header.contentType
            }
            return nil
        }

        var recipient: String? {
            if case let .content(header, _) = state {
                return header.recipient
            }
            return nil
        }

        var channel: String? {
            if case let .content(header, _) = state {
                return header.channel
            }
            return nil
        }

        private var nextRole: Harmony.Role?
        private var state: StreamState

        init(startingRole: Harmony.Role? = nil) {
            nextRole = startingRole
            if startingRole != nil {
                state = .header(headerTokens: [])
            } else {
                state = .expectStart
            }
        }

        mutating func process(_ token: String) throws {
            tokens.append(token)
            try step(with: Harmony.Token(token))
        }

        mutating func processEOS() throws {
            try step(with: nil)
        }

        private mutating func step(with token: Harmony.Token?) throws {
            switch state {
            case .expectStart:
                // NOTE: Ignore EOS while waiting
                guard let token else { return }

                switch token {
                case .special(.start.self):
                    state = .header(headerTokens: [])
                default:
                    throw ParserError.unexpectedToken(token.rawValue)
                }

            case var .header(headerTokens):
                guard let token else {
                    throw ParserError.eosWhileWaitingForHeader
                }
                switch token {
                case .special(.message.self):
                    let headerString = concatenateHeaderTokens(headerTokens)
                    let header = try parseHeader(from: headerString, roleOverride: nextRole)
                    nextRole = nil
                    state = .content(header: header, content: "")
                default:
                    headerTokens.append(token)
                    state = .header(headerTokens: headerTokens)
                }

            case .content(let header, var content):
                let isEOS: Bool
                if let token {
                    switch token {
                    case let .special(s) where s.isStopToken:
                        isEOS = true
                    case let .text(t):
                        content += t
                        delta = t
                        state = .content(header: header, content: content)
                        isEOS = false
                    case let .special(s):
                        // NOTE: Non-stop special tokens in content are treated as text
                        content += s.rawValue
                        delta = s.rawValue
                        state = .content(header: header, content: content)
                        isEOS = false
                    }
                } else {
                    isEOS = true
                }

                if isEOS {
                    let msg = Harmony.Message(
                        author: header.author,
                        recipient: header.recipient,
                        channel: header.channel,
                        contentType: header.contentType,
                        content: [.text(content)]
                    )
                    messages.append(msg)
                    delta = nil
                    state = .expectStart
                }
            }
        }

        private func concatenateHeaderTokens(_ tokens: [Harmony.Token]) -> String {
            var out = ""
            for t in tokens {
                switch t {
                case let .special(s): out += s.rawValue
                case let .text(s): out += s
                }
            }
            return out.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        private func parseHeader(
            from raw: String,
            roleOverride: Harmony.Role?
        ) throws -> ParsedHeader {
            var headerString = raw
            var channel: String? = nil

            if let range = headerString.range(of: Harmony.Token.Special.channel.rawValue) {
                let after = headerString[range.upperBound...]
                let remainder = String(after)
                let channelEndIndex: String.Index = if let idx = remainder
                    .firstIndex(where: { $0.isWhitespace || $0 == "<" })
                {
                    idx
                } else { remainder.endIndex }
                let value = String(remainder[..<channelEndIndex])
                if value
                    .isEmpty
                {
                    throw ParserError.invalidHeader("channel marker present without value")
                }
                channel = value

                // NOTE: Rebuild the header string without the channel segment
                var newHeader = String(headerString[..<range.lowerBound])
                newHeader += String(remainder[channelEndIndex...])
                headerString = newHeader
            }

            headerString = headerString.trimmingCharacters(in: .whitespacesAndNewlines)

            if headerString.contains(Harmony.Token.Special.constrainedFormat.rawValue) {
                headerString = headerString.replacingOccurrences(
                    of: Harmony.Token.Special.constrainedFormat.rawValue,
                    with: " \(Harmony.Token.Special.constrainedFormat.rawValue)"
                )
                .trimmingCharacters(in: .whitespacesAndNewlines)
            }

            var parts = headerString.split(whereSeparator: { $0.isWhitespace }).map(String.init)

            var roleStrOpt: String? = nil
            let role: Harmony.Role = try {
                if let r = roleOverride { return r }
                guard let first = parts.first
                else { throw ParserError.invalidHeader("missing role") }
                roleStrOpt = first
                if let r = Harmony.Role(rawValue: first) { return r }
                // NOTE: If there's an unknown first token, treat it as a tool
                //       name if other parts exist or it looks like a recipient.
                if parts.count > 1 || first.hasPrefix("to=") {
                    parts.removeFirst()
                    return .tool
                }
                throw ParserError.invalidHeader("unknown role: \(first)")
            }()

            if let first = parts.first, first == role.rawValue { parts.removeFirst() }

            var recipient: String? = nil
            var contentType: String? = nil

            if !parts.isEmpty {
                let last = parts.removeLast()
                if last.hasPrefix("to=") {
                    recipient = String(last.dropFirst(3))
                } else if (parts.count + 1) ==
                    1
                { // only one token total after possible role removal
                    recipient = last
                } else {
                    contentType = last
                    if let maybeRecipient = parts.popLast() {
                        recipient = maybeRecipient
                            .hasPrefix("to=") ? String(maybeRecipient.dropFirst(3)) :
                            maybeRecipient
                    }
                }
            }

            // NOTE: Ensure all header tokens were consumed during parsing
            guard parts.isEmpty else {
                throw ParserError.invalidHeader("unexpected tokens remaining: \(parts)")
            }

            let author: Harmony.Author = {
                if role == .tool {
                    return Harmony.Author(role: role, name: roleStrOpt)
                }
                return Harmony.Author(role: role, name: nil)
            }()

            return ParsedHeader(
                author: author,
                recipient: recipient,
                channel: channel,
                contentType: contentType
            )
        }
    }

    enum ParserError: Error, CustomStringConvertible, Equatable, Sendable {
        case unexpectedToken(String)
        case eosWhileWaitingForHeader
        case invalidHeader(String)

        var description: String {
            switch self {
            case let .unexpectedToken(s): s
            case .eosWhileWaitingForHeader: "Unexpected EOS while waiting for header"
            case let .invalidHeader(s): "Invalid header: \(s)"
            }
        }
    }
}

extension Harmony {
    enum Token: CustomStringConvertible, Equatable, Sendable {
        case special(Special)
        case text(String)

        init(_ string: String) {
            if let special = Special(rawValue: string) {
                self = .special(special)
            } else {
                self = .text(string)
            }
        }

        var rawValue: String {
            switch self {
            case let .special(s): s.rawValue
            case let .text(t): t
            }
        }

        var description: String { rawValue }
    }
}

extension Harmony.Token {
    enum Special: String, Equatable, Sendable {
        case start = "<|start|>"
        case message = "<|message|>"
        case endMessage = "<|end|>"
        case endMessageDoneSampling = "<|return|>"
        case endMessageAssistantToTool = "<|call|>"
        case channel = "<|channel|>"
        case constrainedFormat = "<|constrain|>"

        var isStopToken: Bool {
            switch self {
            case .endMessage, .endMessageDoneSampling, .endMessageAssistantToTool:
                true
            case .start, .message, .channel, .constrainedFormat:
                false
            }
        }

        var isToolCall: Bool {
            self == .endMessageAssistantToTool
        }

        static var stopTokens: Set<String> {
            [
                endMessage.rawValue,
                endMessageDoneSampling.rawValue,
                endMessageAssistantToTool.rawValue,
            ]
        }

        var description: String { rawValue }
    }
}

func ~= (lhs: Harmony.Token.Special, rhs: String) -> Bool {
    lhs.rawValue == rhs
}
