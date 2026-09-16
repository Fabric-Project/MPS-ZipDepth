//
//  ZipDepthPreprocessor.swift
//  MPSZipDepth
//

import Foundation
import Metal
import simd

/// Encodes a resample of an arbitrary source texture directly into the
/// tightly-packed NHWC float32 RGB buffer `ZipDepthMPSGraph` requires, with
/// no CPU-side copy.
public final class ZipDepthPreprocessor
{
    private struct Uniforms
    {
        var outputSize: simd_uint2
        var textureTransform: simd_float4x4
    }

    private let pipeline: MTLComputePipelineState

    public init(device: MTLDevice) throws
    {
        guard
            let shaderURL = Bundle.module.url(
                forResource: "ZipDepthCompute",
                withExtension: "metal",
                subdirectory: "Compute"
            ),
            let source = try? String(contentsOf: shaderURL, encoding: .utf8),
            let library = try? device.makeLibrary(source: source, options: nil),
            let function = library.makeFunction(name: "zipDepthPrepareRGB")
        else
        {
            throw ZipDepthError("Could not load the ZipDepth RGB preprocessing kernel")
        }

        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    /// `textureTransform` describes how `inputTexture` maps onto
    /// presentation pixels; pass identity if there's no transform.
    public func encode(
        inputTexture: MTLTexture,
        textureTransform: simd_float4x4,
        outputBuffer: MTLBuffer,
        outputWidth: Int,
        outputHeight: Int,
        commandBuffer: MTLCommandBuffer
    ) throws
    {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else
        {
            throw ZipDepthError("Could not create the ZipDepth RGB preprocess encoder")
        }

        var uniforms = Uniforms(
            outputSize: simd_uint2(UInt32(outputWidth), UInt32(outputHeight)),
            textureTransform: textureTransform
        )

        encoder.label = "ZipDepth RGB Preprocess"
        encoder.setComputePipelineState(self.pipeline)
        encoder.setTexture(inputTexture, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 0)
        encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)

        Self.dispatch(encoder, pipeline: self.pipeline, width: outputWidth, height: outputHeight)
        encoder.endEncoding()
    }

    private static func dispatch(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        width: Int,
        height: Int
    )
    {
        let threadWidth = pipeline.threadExecutionWidth
        let threadHeight = max(1, pipeline.maxTotalThreadsPerThreadgroup / threadWidth)
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        )
    }
}
