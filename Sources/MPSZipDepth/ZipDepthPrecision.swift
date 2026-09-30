import MetalPerformanceShadersGraph

/// The precision ZipDepth's graph computes in. Inputs and outputs stay
/// float32 in every mode.
public enum ZipDepthPrecision: Sendable, Equatable
{
    /// Float32 throughout.
    case float32
    /// Convolutions in float16 (weights and arithmetic), everything between
    /// them float32.
    case mixedFloat16
    /// Float16 throughout, with one cast after the input and one before the
    /// output.
    case float16

    /// The type activations flow through between layers.
    var activationDataType: MPSDataType { self == .float16 ? .float16 : .float32 }

    /// The type convolutions run in.
    var layerDataType: MPSDataType { self == .float32 ? .float32 : .float16 }
}
