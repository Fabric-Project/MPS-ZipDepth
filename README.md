# MPS-ZipDepth

A Swift Package Manager implementation of
[ZipDepth](https://github.com/fabiotosi92/ZipDepth) using Metal Performance
Shaders Graph directly. It does not use Core ML, ONNX Runtime, PyTorch, or any
third-party runtime.

The package implements the official `base` architecture with the official NPU
checkpoint and its unfold-free learned upsampling head. The graph includes the
RepVGG branches, strip pooling, multi-scale depthwise convolutions,
squeeze-and-excitation, global context, SPPF, cross-scale fusion, FPN decoder,
and learned nearest/bilinear upsampling blend.

## Requirements

- macOS 15, iOS 18, or visionOS 2
- Swift 5.9 or newer
- A Metal device

## Usage

```swift
import Metal
import MPSZipDepth

guard let device = MTLCreateSystemDefaultDevice(),
      let commandQueue = device.makeCommandQueue() else {
    return
}

let zipDepth = try ZipDepthMPSGraph(
    inputWidth: 384,
    inputHeight: 384,
    commandQueue: commandQueue
)

// NHWC, tightly packed float32 RGB in the 0...1 range.
let depth: [Float] = try zipDepth.run(inputBuffer: rgbBuffer)
```

Input width and height may vary independently but must be multiples of 32. A
model instance is compiled for one input shape. The output has the same width
and height and contains nonnegative relative-depth values, not metric distance.

The execution API mirrors MPS-MediaPipe:

- `run(inputBuffer:)` is the synchronous path: it creates its own command
  buffer, commits it, waits, and returns the depth values.
- `submit(inputBuffer:commandBuffer:completion:)` is the asynchronous path:
  it encodes onto the caller's `MPSCommandBuffer` and delivers the depth
  values to `completion` after the caller commits and the GPU finishes.
- `encode(inputBuffer:outputBuffer:commandBuffer:)` is the GPU-resident,
  no-CPU-readback path.

`submit` and `encode` take the caller's own `MPSCommandBuffer` and never wrap,
commit, or wait on it; the caller commits it. MPSGraph may call
`commitAndContinue()` internally, and the caller's persistent wrapper is what
follows the replacement root buffer, so preprocessing, inference, and
dependent postprocessing stay ordered on the GPU without a CPU wait. Both
return `false` instead of blocking when every in-flight slot is occupied.

`prediction(rgb:)` remains as a convenience path for callers without a GPU
preprocessing buffer. Resize/crop, color conversion, and NHWC packing should
happen upstream on the GPU in real-time pipelines.

## Weights

`Scripts/export_zipdepth_weights.py` converts the official PyTorch checkpoint
to a flat float32 binary and JSON shape manifest without requiring PyTorch. The
bundled files were generated from `zipdepth_base_npu.pth` at upstream commit
`91f3fd21e131641f51e8d35736d1958350180e3a`:

```sh
python3 Scripts/export_zipdepth_weights.py \
  zipdepth_base_npu.pth \
  Sources/MPSZipDepth/Models/ZipDepthBaseNPU_weights.bin \
  Sources/MPSZipDepth/Models/ZipDepthBaseNPU_weights.json
```

The official implementation and pretrained checkpoint are MIT licensed. See
`LICENSE-ZipDepth` for the upstream license notice.

## Validation

```sh
swift test
```

Tests compile and execute the complete MPSGraph with the bundled official
weights, and verify input contracts and finite, nonnegative output.
