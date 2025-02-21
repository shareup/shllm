import Testing
@testable import SHLLM

@Test func example() async throws {
    let yo = SHLLM()
    try await yo.hello()
}
