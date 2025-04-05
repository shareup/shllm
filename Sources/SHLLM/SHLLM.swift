import CoreGraphics
import Foundation
import Metal
import MLX

public enum SHLLM {
    public static func clearCache() {
        MLX.GPU.clearCache()
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

@_exported import struct MLXLLM.Qwen2Configuration
@_exported import class MLXLLM.Qwen2Model

@_exported import struct MLXLLM.GemmaConfiguration
@_exported import class MLXLLM.GemmaModel

@_exported import struct MLXLLM.Gemma2Configuration
@_exported import class MLXLLM.Gemma2Model

@_exported import struct MLXLLM.LlamaConfiguration
@_exported import class MLXLLM.LlamaModel

@_exported import struct MLXLLM.OpenElmConfiguration
@_exported import class MLXLLM.OpenELMModel

@_exported import struct MLXLLM.PhiConfiguration
@_exported import class MLXLLM.PhiModel

@_exported import struct MLXLLM.Phi3Configuration
@_exported import class MLXLLM.Phi3Model

@_exported import struct MLXLLM.PhiMoEConfiguration
@_exported import class MLXLLM.PhiMoEModel
