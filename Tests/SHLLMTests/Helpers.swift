import Foundation
import SHLLM

func loadModel<M: InitializableWithDirectory>(
    from directory: @autoclosure () throws -> URL
) async throws -> M? {
    #if targetEnvironment(simulator)
        Swift.print("⚠️ LLMs are not supported in the Simulator")
        return nil
    #else
        do {
            return try await M(directory: directory())
        } catch let SHLLMError.directoryNotFound(name) {
            Swift.print("⚠️ \(name) does not exist")
            return nil
        } catch let SHLLMError.missingBundle(name) {
            Swift.print("⚠️ \(name) bundle does not exist")
            return nil
        } catch {
            throw error
        }
    #endif
}

protocol InitializableWithDirectory {
    init(directory: URL) async throws
}
