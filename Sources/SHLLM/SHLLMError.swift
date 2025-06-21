import Foundation

public enum SHLLMError: Error, Hashable {
    case directoryNotFound(String)
    case inputTooLong
    case invalidOrMissingConfig(String)
    case missingBundle(String)
    case unimplemented
    case unsupportedDevice
}
