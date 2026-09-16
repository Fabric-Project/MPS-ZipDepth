//
//  ZipDepthCompute.metal
//  MPSZipDepth
//
//  Packs an arbitrary source texture into the tightly-packed NHWC float32
//  RGB buffer ZipDepthMPSGraph requires, and unpacks its row-major float32
//  depth output back into a full-resolution single-channel texture. These
//  are implementation details of ZipDepthMPSGraph's own tensor contract, so
//  they live in this package rather than in a host app's shader directory.
//

#include <metal_stdlib>
using namespace metal;

struct ZipDepthPreprocessUniforms
{
    uint2 outputSize;
    float4x4 textureTransform;
};

struct ZipDepthPostprocessUniforms
{
    uint2 modelSize;
    uint2 outputSize;
};

kernel void zipDepthPrepareRGB(
    texture2d<float, access::sample> inputTexture [[texture(0)]],
    device float *outputRGB [[buffer(0)]],
    constant ZipDepthPreprocessUniforms &uniforms [[buffer(1)]],
    uint2 position [[thread_position_in_grid]])
{
    if (any(position >= uniforms.outputSize))
    {
        return;
    }

    constexpr sampler linearSampler(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 canonicalCoordinate = (float2(position) + 0.5) / float2(uniforms.outputSize);
    float2 storedCoordinate = (uniforms.textureTransform * float4(canonicalCoordinate, 0.0, 1.0)).xy;
    float3 rgb = inputTexture.sample(linearSampler, storedCoordinate).rgb;
    uint outputIndex = (position.y * uniforms.outputSize.x + position.x) * 3;
    outputRGB[outputIndex] = rgb.r;
    outputRGB[outputIndex + 1] = rgb.g;
    outputRGB[outputIndex + 2] = rgb.b;
}

kernel void zipDepthWriteDepth(
    device const float *modelDepth [[buffer(0)]],
    constant ZipDepthPostprocessUniforms &uniforms [[buffer(1)]],
    texture2d<float, access::write> outputTexture [[texture(0)]],
    uint2 position [[thread_position_in_grid]])
{
    if (any(position >= uniforms.outputSize))
    {
        return;
    }

    float2 sourcePosition = (float2(position) + 0.5)
        * float2(uniforms.modelSize) / float2(uniforms.outputSize) - 0.5;
    int2 sourceMinimum = int2(floor(sourcePosition));
    float2 interpolation = fract(sourcePosition);
    uint2 maximumPosition = uniforms.modelSize - 1;
    uint2 topLeft = min(uint2(max(sourceMinimum, int2(0))), maximumPosition);
    uint2 bottomRight = min(uint2(max(sourceMinimum + 1, int2(0))), maximumPosition);
    uint2 topRight = uint2(bottomRight.x, topLeft.y);
    uint2 bottomLeft = uint2(topLeft.x, bottomRight.y);

    float topLeftDepth = modelDepth[topLeft.y * uniforms.modelSize.x + topLeft.x];
    float topRightDepth = modelDepth[topRight.y * uniforms.modelSize.x + topRight.x];
    float bottomLeftDepth = modelDepth[bottomLeft.y * uniforms.modelSize.x + bottomLeft.x];
    float bottomRightDepth = modelDepth[bottomRight.y * uniforms.modelSize.x + bottomRight.x];
    float topDepth = mix(topLeftDepth, topRightDepth, interpolation.x);
    float bottomDepth = mix(bottomLeftDepth, bottomRightDepth, interpolation.x);
    float depth = mix(topDepth, bottomDepth, interpolation.y);

    outputTexture.write(float4(depth, depth, depth, 1.0), position);
}
