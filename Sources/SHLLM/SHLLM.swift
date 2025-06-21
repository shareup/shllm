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

@_exported import class MLXLLM.Gemma2Model
@_exported import class MLXLLM.GemmaModel
@_exported import class MLXLLM.LlamaModel
@_exported import class MLXLLM.OpenELMModel
@_exported import class MLXLLM.Phi3Model
@_exported import class MLXLLM.PhiModel
@_exported import class MLXLLM.PhiMoEModel
@_exported import class MLXLLM.Qwen2Model
@_exported import class MLXLLM.Qwen3Model
@_exported import class MLXLLM.Qwen3MoEModel
