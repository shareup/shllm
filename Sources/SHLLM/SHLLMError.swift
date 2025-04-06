import Foundation

public enum SHLLMError: Error, Hashable {
    case directoryNotFound(String)
    case invalidOrMissingConfig(String)
    case missingBundle(String)
    case unimplemented
    case unsupportedDevice
}
