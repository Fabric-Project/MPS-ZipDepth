//
//  ZipDepthPostprocessor.swift
//  MPSZipDepth
//

import Foundation
import Metal
import simd

/// Encodes a bilinear upsample of `ZipDepthMPSGraph`'s row-major float32
/// depth output directly into a full-resolution single-channel texture, with
/// no CPU-side copy.
public final class ZipDepthPostprocessor
{
    private struct Uniforms
    {
        var modelSize: simd_uint2
        var outputSize: simd_uint2
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
            let function = library.makeFunction(name: "zipDepthWriteDepth")
        else
        {
            throw ZipDepthError("Could not load the ZipDepth depth postprocessing kernel")
        }

        self.pipeline = try device.makeComputePipelineState(function: function)
    }

    public func encode(
        inputBuffer: MTLBuffer,
        modelWidth: Int,
        modelHeight: Int,
        outputTexture: MTLTexture,
        outputWidth: Int,
        outputHeight: Int,
        commandBuffer: MTLCommandBuffer
    ) throws
    {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else
        {
            throw ZipDepthError("Could not create the ZipDepth depth postprocess encoder")
        }

        var uniforms = Uniforms(
            modelSize: simd_uint2(UInt32(modelWidth), UInt32(modelHeight)),
            outputSize: simd_uint2(UInt32(outputWidth), UInt32(outputHeight))
        )

        encoder.label = "ZipDepth Output"
        encoder.setComputePipelineState(self.pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setTexture(outputTexture, index: 0)

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
