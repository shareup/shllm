import CoreGraphics
import Foundation
import Metal
import MLX

public enum SHLLM {
    public static func clearCache() {
        MLX.GPU.clearCache()
    }

    public static var cacheLimit: Int {
        get {
            MLX.GPU.cacheLimit
        }
        set {
            MLX.GPU.set(cacheLimit: newValue)
        }
    }

    public static var memoryLimit: Int {
        get {
            MLX.GPU.memoryLimit
        }
        set {
            MLX.GPU.set(memoryLimit: newValue)
        }
    }

    public static func withDefaultDevice<R>(
        _ device: MLX.DeviceType,
        _ body: () throws -> R
    ) rethrows -> R {
        switch device {
        case .cpu:
            try MLX.Device.withDefaultDevice(.cpu, body)
        case .gpu:
            try MLX.Device.withDefaultDevice(.gpu, body)
        }
    }

    public static func withDefaultDevice<R>(
        _ device: MLX.DeviceType,
        _ body: () async throws -> R
    ) async rethrows -> R {
        switch device {
        case .cpu:
            try await MLX.Device.withDefaultDevice(.cpu, body)
        case .gpu:
            try await MLX.Device.withDefaultDevice(.gpu, body)
        }
    }

    public static var recommendedMaxWorkingSetSize: Int {
        guard let device = MTLCreateSystemDefaultDevice(),
              device.recommendedMaxWorkingSetSize < Int.max
        else { return 0 }
        return Int(device.recommendedMaxWorkingSetSize)
    }

    public static var isSupportedDevice: Bool {
        guard let _ = MTLCreateSystemDefaultDevice() else {
            return false
        }
        return true
    }

    static var assertSupportedDevice: Void {
        get throws {
            guard isSupportedDevice else {
                throw SHLLMError.unsupportedDevice
            }
        }
    }
}

@_exported import enum MLXLMCommon.Chat
@_exported import struct MLXLMCommon.Message
@_exported import struct MLXLMCommon.UserInput

@_exported import struct MLXLMCommon.Tool
@_exported import struct MLXLMCommon.ToolParameter
@_exported import enum MLXLMCommon.ToolParameterType
@_exported import protocol MLXLMCommon.ToolProtocol

extension Chat.Message: @retroactive @unchecked Sendable {}

@_exported import enum MLX.DeviceType

@_exported import protocol MLXLMCommon.LanguageModel

@_exported import class MLXLLM.Gemma2Model
@_exported import class MLXLLM.Gemma3TextModel
@_exported import class MLXLLM.GemmaModel
@_exported import class MLXLLM.LlamaModel
@_exported import protocol MLXLLM.LLMModel
@_exported import class MLXLLM.OpenELMModel
@_exported import class MLXLLM.Phi3Model
@_exported import class MLXLLM.PhiModel
@_exported import class MLXLLM.PhiMoEModel
@_exported import class MLXLLM.Qwen2Model
@_exported import class MLXLLM.Qwen3Model
@_exported import class MLXLLM.Qwen3MoEModel

@_exported import class MLXVLM.Gemma3
@_exported import protocol MLXVLM.VLMModel
