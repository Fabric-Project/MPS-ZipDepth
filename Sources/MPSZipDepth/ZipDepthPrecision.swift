import Foundation

/// The data type ZipDepth's graph computes in. Inputs and outputs stay
/// float32 either way; `float16` casts once after the input and once before
/// the output, with every op and constant in between in float16.
public enum ZipDepthPrecision: Sendable, Equatable
{
    case float32
    case float16
}
