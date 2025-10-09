import Foundation
import MLXLMCommon

public actor ModelCache {
    private var cachedContext: (key: String, context: ModelContext)?

    public init() {}

    public func getOrLoadModel(
        type _: (some LanguageModel).Type,
        directory: URL,
        customConfiguration: ((ModelConfiguration) -> ModelConfiguration)?
    ) async throws -> ModelContext {
        let key = directory.path

        if let cached = cachedContext, cached.key == key {
            return cached.context
        }

        if cachedContext != nil {
            cachedContext = nil
        }

        try SHLLM.assertSupportedDevice
        let factory = try await loadModel(directory: directory)

        let config = customConfiguration?(factory.configuration) ?? factory.configuration

        let context = ModelContext(
            configuration: config,
            model: factory.model,
            processor: factory.processor,
            tokenizer: factory.tokenizer
        )

        cachedContext = (key: key, context: context)
        return context
    }

    public func isModelCached(directory: URL) -> Bool {
        cachedContext?.key == directory.path
    }

    public func clearCache() {
        cachedContext = nil
    }
}
