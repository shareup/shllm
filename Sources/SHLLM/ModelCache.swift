import AsyncExtensions
import Foundation
import MLXLMCommon

struct CachedModel {
    let directory: URL
    let contex: ModelContext
}

struct Loader {
    let future: AsyncThrowingFuture<ModelContext>
    let work: () async throws -> CachedModel
    var task: Task<CachedModel, Error>?
}

actor ModelCache {
    var currentCache: CachedModel?
    var isWorking = false
    var loaders: [Loader] = []

    func next() async throws {
        if isWorking { return }

        guard !loaders.isEmpty else {
            return
        }

        var loader = loaders.removeFirst()

        isWorking = true
        defer { isWorking = false }

        loader.task = Task {
            try await loader.work()
        }

        let cached = try await loader.task!.value
        loader.future.resolve(cached.contex)

        try await next()
    }

    func clear() {
        for loader in loaders {
            loader.future.fail(CancellationError())
            loader.task?.cancel()
        }
        currentCache = nil
    }

    func getOrLoadModel(
        directory: URL,
        maxInputTokenCount: Int?,
        customConfiguration: ((ModelConfiguration) -> ModelConfiguration)?
    ) async throws -> AsyncThrowingFuture<ModelContext> {
        let future = AsyncThrowingFuture<ModelContext>()

        let work = { () async throws in
            try await self.realGetOrLoadModel(
                directory: directory,
                maxInputTokenCount: maxInputTokenCount,
                customConfiguration: customConfiguration
            )
        }

        loaders.append(Loader(future: future, work: work))

        if loaders.count == 1 {
            Task {
                try await self.next()
            }
        }

        return future
    }

    func realGetOrLoadModel(
        directory: URL,
        maxInputTokenCount: Int?,
        customConfiguration: ((ModelConfiguration) -> ModelConfiguration)?
    ) async throws -> CachedModel {
        let model: CachedModel

        if let currentCache, currentCache.directory == directory {
            model = currentCache
        } else {
            try SHLLM.assertSupportedDevice
            let baseContext = try await loadModel(directory: directory)

            let config = customConfiguration?(baseContext.configuration) ?? baseContext
                .configuration

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

            model = CachedModel(directory: directory, contex: context)
        }

        currentCache = model
        return model
    }
}
