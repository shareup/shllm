@testable import SHLLM
import Testing

@Test
func canBuildLibrary() async throws {
    #expect(Bool(true))
}

@Test
func cacheLimit() async throws {
    let limit = 1024
    #expect(SHLLM.cacheLimit != limit)
    SHLLM.cacheLimit = limit
    #expect(SHLLM.cacheLimit == limit)
}

@Test
func memoryLimit() async throws {
    let limit = 1024
    #expect(SHLLM.memoryLimit != limit)
    SHLLM.memoryLimit = limit
    #expect(SHLLM.memoryLimit == limit)
}

@Test
func recommendedMaxWorkingSetSize() async throws {
    let recommended = SHLLM.recommendedMaxWorkingSetSize
    #expect(recommended > 0)
}
