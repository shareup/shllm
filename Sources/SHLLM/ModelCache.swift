import Foundation
import MLXLMCommon
import Synchronized

func loadModelContext(
    directory: URL,
    maxInputTokenCount: Int?,
    customConfiguration: ((ModelConfiguration) -> ModelConfiguration)?,
    forceClearCache: Bool = false
) async throws -> ModelContext {
    let model = cache.access { modelCache -> CachedModel? in
        if forceClearCache {
            print(
                "🔄 SHLLM_MODEL_CACHE: force clearing cache for \(directory.lastPathComponent)"
            )
            modelCache.clear()
            return nil
        }

        guard let model = modelCache.matching(directory: directory) else {
            // NOTE: Since we're going to be replacing this cached model,
            //       we may as well release our reference to it. If a client
            //       is still using it, their strong reference to the model
            //       will keep it alive as long as needed.
            print(
                "🔄 SHLLM_MODEL_CACHE: cache miss, will load model from \(directory.lastPathComponent)"
            )
            modelCache.clear()
            return nil
        }
        print("✅ SHLLM_MODEL_CACHE: cache hit for \(directory.lastPathComponent)")
        return model
    }

    if let model {
        return model.context
    } else {
        print("📦 SHLLM_MODEL_CACHE: loading model from disk: \(directory.lastPathComponent)")
        let loadStart = Date()
        try SHLLM.assertSupportedDevice
        let baseContext = try await loadModel(directory: directory)

        let config = customConfiguration?(baseContext.configuration)
            ?? baseContext.configuration

        let processor = TruncatingUserInputProcessor(
            wrapping: baseContext.processor,
            tokenizer: baseContext.tokenizer,
            maxInputTokenCount: maxInputTokenCount
        )

        let context = ModelContext(
            configuration: config,
            model: baseContext.model,
            processor: processor,
            tokenizer: baseContext.tokenizer
        )

        let model = CachedModel(
            directory: directory,
            context: context
        )
        cache.access { $0.replace(with: model) }

        let loadTime = Date().timeIntervalSince(loadStart)
        print("✅ SHLLM_MODEL_CACHE: model loaded in \(String(format: "%.2f", loadTime))s")

        return context
    }
}

enum ModelCache {
    static var isEnabled: Bool {
        cache.access { cache in
            switch cache {
            case .disabled: false
            case .enabled: true
            }
        }
    }

    static func enable() {
        cache.access { $0.enable() }
    }

    static func disable() {
        cache.access { $0.disable() }
    }

    static func clear() {
        cache.access { $0.clear() }
    }
}

private let cache = Locked<Cache>(.enabled(nil))

private struct CachedModel {
    let directory: URL
    let context: ModelContext
}

private enum Cache {
    case disabled
    case enabled(CachedModel?)

    mutating func enable() {
        switch self {
        case .disabled: self = .enabled(nil)
        case .enabled: break
        }
    }

    mutating func disable() {
        self = .disabled
    }

    func matching(directory: URL) -> CachedModel? {
        guard case let .enabled(cachedModel) = self,
              cachedModel?.directory == directory
        else { return nil }
        return cachedModel
    }

    mutating func replace(with model: CachedModel) {
        switch self {
        case .disabled: break
        case .enabled: self = .enabled(model)
        }
    }

    mutating func clear() {
        switch self {
        case .disabled: break
        case .enabled: self = .enabled(nil)
        }
    }
}
