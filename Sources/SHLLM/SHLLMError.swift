import Foundation

public enum SHLLMError: Error {
    case directoryNotFound(String)
    case invalidOrMissingConfig(String)
}
