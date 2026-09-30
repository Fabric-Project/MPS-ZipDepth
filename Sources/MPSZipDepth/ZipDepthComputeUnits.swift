import MetalPerformanceShadersGraph

/// Where a ZipDepth graph may run. MPSGraph can be allowed to use the Neural
/// Engine but cannot be forced onto it; in practice only float16 graphs with
/// enough depthwise work reach it.
public enum ZipDepthComputeUnits: Sendable, Equatable
{
    /// MPSGraph's placement pass may move work to the Neural Engine or CPU.
    case gpuAndNeuralEngine
    /// Everything stays on the GPU.
    case gpuOnly

    /// `.level1` runs MPSGraph's placement pass; `.level0` skips it (and
    /// MPSGraph's other level-1 optimizations).
    var optimizationLevel: MPSGraphOptimization
    {
        switch self
        {
        case .gpuAndNeuralEngine: .level1
        case .gpuOnly: .level0
        }
    }
}
