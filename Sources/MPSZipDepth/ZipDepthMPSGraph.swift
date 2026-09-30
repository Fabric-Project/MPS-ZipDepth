import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// The official ZipDepth-base NPU graph, implemented directly with MPSGraph.
///
/// Input is one NHWC, tightly packed, float32 RGB image in the 0...1 range.
/// Output is one full-resolution float32 depth map in row-major order. ZipDepth
/// predicts relative depth; values are not metric distances.
public final class ZipDepthMPSGraph
{
    public let inputWidth: Int
    public let inputHeight: Int

    public var inputBufferLength: Int
    {
        self.inputWidth * self.inputHeight * 3 * MemoryLayout<Float>.stride
    }

    public var outputBufferLength: Int
    {
        self.inputWidth * self.inputHeight * MemoryLayout<Float>.stride
    }

    private let graph = MPSGraph()
    private let commandQueue: MTLCommandQueue
    private let device: MPSGraphDevice
    private let inputTensor: MPSGraphTensor
    private let outputTensor: MPSGraphTensor
    private let executable: MPSGraphExecutable
    private let slotSemaphore: DispatchSemaphore
    private let slotLock = NSLock()
    private var freeSlots: [Int]
    private let outputBufferCache: [OutputBufferSlot]

    private final class OutputBufferSlot
    {
        var values: [Float] = []
    }

    public init(
        weightsBinaryURL: URL,
        weightsManifestURL: URL,
        inputWidth: Int,
        inputHeight: Int,
        commandQueue: MTLCommandQueue,
        maxFramesInFlight: Int = 3,
        precision: ZipDepthPrecision = .float32,
        computeUnits: ZipDepthComputeUnits = .gpuAndNeuralEngine
    ) throws
    {
        guard inputWidth > 0, inputHeight > 0,
              inputWidth.isMultiple(of: 32), inputHeight.isMultiple(of: 32) else
        {
            throw ZipDepthError("ZipDepth input dimensions must be positive multiples of 32.")
        }
        guard maxFramesInFlight > 0 else
        {
            throw ZipDepthError("ZipDepth maxFramesInFlight must be positive.")
        }

        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.commandQueue = commandQueue
        self.device = MPSGraphDevice(mtlDevice: commandQueue.device)
        self.slotSemaphore = DispatchSemaphore(value: maxFramesInFlight)
        self.freeSlots = Array(0..<maxFramesInFlight)
        self.outputBufferCache = (0..<maxFramesInFlight).map { _ in OutputBufferSlot() }

        let weights = try ZipDepthWeights(
            binaryURL: weightsBinaryURL,
            manifestURL: weightsManifestURL
        )
        let builder = GraphBuilder(
            graph: self.graph,
            weights: weights,
            inputWidth: inputWidth,
            inputHeight: inputHeight,
            dataType: precision.activationDataType,
            layerDataType: precision.layerDataType
        )

        let input = self.graph.placeholder(
            shape: [1, NSNumber(value: inputHeight), NSNumber(value: inputWidth), 3],
            dataType: .float32,
            name: "rgb"
        )
        self.inputTensor = input
        self.outputTensor = try builder.build(inputNHWC: input)

        let inputType = MPSGraphShapedType(shape: input.shape ?? [], dataType: .float32)
        let descriptor = MPSGraphCompilationDescriptor()
        descriptor.optimizationLevel = computeUnits.optimizationLevel
        descriptor.waitForCompilationCompletion = true
        if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *)
        {
            descriptor.reducedPrecisionFastMath = .allowFP16Intermediates
        }
        self.executable = self.graph.compile(
            with: self.device,
            feeds: [input: inputType],
            targetTensors: [self.outputTensor],
            targetOperations: nil,
            compilationDescriptor: descriptor
        )
        self.executable.specialize(
            with: self.device,
            inputTypes: [inputType],
            compilationDescriptor: descriptor
        )
    }

    /// Synchronous MPS-MediaPipe-style inference entry point.
    public func run(inputBuffer: MTLBuffer) throws -> [Float]
    {
        let requiredBytes = self.inputBufferLength
        guard inputBuffer.length >= requiredBytes else
        {
            throw ZipDepthError("Input buffer has \(inputBuffer.length) bytes; ZipDepth requires \(requiredBytes).")
        }

        let slot = self.acquireSlotBlocking()
        defer { self.releaseSlot(slot) }

        let inputData = MPSGraphTensorData(
            inputBuffer,
            shape: self.inputTensor.shape ?? [],
            dataType: .float32
        )
        guard let result = self.executable.run(
            with: self.commandQueue,
            inputs: [inputData],
            results: nil,
            executionDescriptor: nil
        ).first else
        {
            throw ZipDepthError("ZipDepth produced no output tensor.")
        }

        return self.floatArray(from: result, slot: slot)
    }

    /// Asynchronous inference: encodes onto the caller's `MPSCommandBuffer`
    /// and calls `completion` with the depth values once that work finishes
    /// on the GPU. Never commits or waits -- the caller owns `commandBuffer`
    /// and commits it, and `completion` fires only after that. Throws for a
    /// genuinely invalid call (wrong-sized buffer, mismatched device); returns
    /// `false` (never calls `completion`) only when every `maxFramesInFlight`
    /// slot is busy, which is ordinary backpressure a caller retries next frame.
    @discardableResult
    public func submit(
        inputBuffer: MTLBuffer,
        commandBuffer: MPSCommandBuffer,
        completion: @escaping (Result<[Float], any Error>) -> Void
    ) throws -> Bool
    {
        guard inputBuffer.length >= self.inputBufferLength else
        {
            throw ZipDepthError("Input buffer has \(inputBuffer.length) bytes; ZipDepth requires \(self.inputBufferLength).")
        }
        guard commandBuffer.device === self.commandQueue.device else
        {
            throw ZipDepthError("The command buffer and ZipDepth model use different Metal devices.")
        }
        guard let slot = self.acquireSlotNonBlocking() else
        {
            return false
        }

        let inputData = MPSGraphTensorData(
            inputBuffer,
            shape: self.inputTensor.shape ?? [],
            dataType: .float32
        )
        let executionDescriptor = MPSGraphExecutableExecutionDescriptor()
        executionDescriptor.waitUntilCompleted = false
        executionDescriptor.completionHandler = { [weak self] results, error in
            guard let self else { return }
            defer { self.releaseSlot(slot) }

            if let error
            {
                completion(.failure(error))
            }
            else if let result = results.first
            {
                completion(.success(self.floatArray(from: result, slot: slot)))
            }
            else
            {
                completion(.failure(ZipDepthError("ZipDepth produced no output tensor.")))
            }
        }

        _ = self.executable.encode(
            to: commandBuffer,
            inputs: [inputData],
            results: nil,
            executionDescriptor: executionDescriptor
        )
        return true
    }

    @available(*, deprecated, renamed: "run(inputBuffer:)")
    public func prediction(inputBuffer: MTLBuffer) throws -> [Float]
    {
        try self.run(inputBuffer: inputBuffer)
    }

    /// GPU-resident inference: encodes onto the caller's `MPSCommandBuffer`,
    /// writing depth into `outputBuffer`, with no CPU readback. Never commits
    /// or waits -- the caller owns `commandBuffer` and commits it.
    ///
    /// `commandBuffer` must be the caller's own persistent `MPSCommandBuffer`
    /// instance: `MPSGraphExecutable.encode(to:)` may `commitAndContinue` it
    /// internally, and only that one instance follows the continuation, so
    /// the caller's later `commit()` on it is always correct. Work encoded
    /// after this call on the same `commandBuffer`, or on a later command
    /// buffer committed to the same queue, reads `outputBuffer` with no wait.
    ///
    /// Returns `false` (encodes nothing) when every `maxFramesInFlight` slot
    /// is busy -- a dropped frame, not a CPU stall. The slot is released when
    /// the graph's final encoded segment completes.
    @discardableResult
    public func encode(
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        commandBuffer: MPSCommandBuffer
    ) throws -> Bool
    {
        guard commandBuffer.device === self.commandQueue.device else
        {
            throw ZipDepthError("The command buffer and ZipDepth model use different Metal devices.")
        }
        guard inputBuffer.length >= self.inputBufferLength else
        {
            throw ZipDepthError("Input buffer has \(inputBuffer.length) bytes; ZipDepth requires \(self.inputBufferLength).")
        }
        guard outputBuffer.length >= self.outputBufferLength else
        {
            throw ZipDepthError("Output buffer has \(outputBuffer.length) bytes; ZipDepth requires \(self.outputBufferLength).")
        }
        guard let slot = self.acquireSlotNonBlocking() else { return false }

        let inputData = MPSGraphTensorData(
            inputBuffer,
            shape: self.inputTensor.shape ?? [],
            dataType: .float32
        )
        let outputData = MPSGraphTensorData(
            outputBuffer,
            shape: self.outputTensor.shape ?? [],
            dataType: .float32
        )

        let executionDescriptor = MPSGraphExecutableExecutionDescriptor()
        executionDescriptor.waitUntilCompleted = false

        _ = self.executable.encode(
            to: commandBuffer,
            inputs: [inputData],
            results: [outputData],
            executionDescriptor: executionDescriptor
        )
        // Added after MPSGraph finishes encoding: it may have called
        // commitAndContinue, and attaching to the persistent wrapper now
        // targets its current live root, so the slot cannot be released
        // before the graph's final segment completes.
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseSlot(slot)
        }
        return true
    }

    /// Convenience path for callers without an existing GPU preprocessing buffer.
    public func prediction(rgb: [Float]) throws -> [Float]
    {
        let requiredCount = self.inputWidth * self.inputHeight * 3
        guard rgb.count == requiredCount else
        {
            throw ZipDepthError("RGB array has \(rgb.count) values; ZipDepth requires \(requiredCount).")
        }
        guard let buffer = self.commandQueue.device.makeBuffer(
            bytes: rgb,
            length: rgb.count * MemoryLayout<Float>.stride
        ) else
        {
            throw ZipDepthError("Could not allocate the ZipDepth input buffer.")
        }
        return try self.run(inputBuffer: buffer)
    }

    private func acquireSlotBlocking() -> Int
    {
        self.slotSemaphore.wait()
        self.slotLock.lock()
        defer { self.slotLock.unlock() }
        return self.freeSlots.removeLast()
    }

    private func acquireSlotNonBlocking() -> Int?
    {
        guard self.slotSemaphore.wait(timeout: .now()) == .success else { return nil }
        self.slotLock.lock()
        defer { self.slotLock.unlock() }
        return self.freeSlots.removeLast()
    }

    private func releaseSlot(_ slot: Int)
    {
        self.slotLock.lock()
        self.freeSlots.append(slot)
        self.slotLock.unlock()
        self.slotSemaphore.signal()
    }

    private func floatArray(from tensorData: MPSGraphTensorData, slot: Int) -> [Float]
    {
        let count = self.inputWidth * self.inputHeight
        let cache = self.outputBufferCache[slot]
        if cache.values.count != count
        {
            cache.values = [Float](repeating: 0, count: count)
        }
        cache.values.withUnsafeMutableBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            tensorData.mpsndarray().readBytes(baseAddress, strideBytes: nil)
        }
        return cache.values
    }
}

private struct GraphBuilder
{
    let graph: MPSGraph
    let weights: ZipDepthWeights
    let inputWidth: Int
    let inputHeight: Int
    /// The type activations and their constants use; see ZipDepthPrecision.
    let dataType: MPSDataType
    /// The type convolutions run in (float16 for mixedFloat16 and float16).
    let layerDataType: MPSDataType

    func build(inputNHWC: MPSGraphTensor) throws -> MPSGraphTensor
    {
        let typedInput = inputNHWC.dataType == self.dataType ? inputNHWC : self.graph.cast(inputNHWC, to: self.dataType, name: nil)
        var input = self.graph.transpose(typedInput, permutation: [0, 3, 1, 2], name: "input_nchw")
        let mean = try self.broadcastConstant(named: "mean")
        // Multiply by 1/std rather than divide: MPSGraph places float16
        // graphs on the Neural Engine, whose compiler rejects elementwise
        // divides and silently falls the whole graph back to the GPU.
        let inverseStandardDeviation = self.channelConstant(try self.weights.floats(named: "std").map { 1 / $0 })
        input = self.graph.multiplication(
            self.graph.subtraction(input, mean, name: nil),
            inverseStandardDeviation,
            name: "normalized_input"
        )

        let stemHalf = try self.convBN(input, prefix: "encoder.stem_half", stride: 2)
        let stemQuarter = try self.convBN(stemHalf, prefix: "encoder.stem_quarter", stride: 2)

        var stage1 = stemQuarter
        for index in 0..<2
        {
            stage1 = try self.qaRepBlock(stage1, prefix: "encoder.stage1.\(index)", identity: true)
        }

        var stage2 = try self.qaRepBlock(stage1, prefix: "encoder.down2", stride: 2, identity: false)
        for index in 0..<2
        {
            stage2 = try self.qaRepBlock(stage2, prefix: "encoder.stage2.\(index)", identity: true)
        }
        stage2 = try self.minimalMultiScale(stage2, prefix: "encoder.stage2.2")
        stage2 = try self.stripPoolingAttention(stage2, prefix: "encoder.stage2.3")

        var stage3 = try self.qaRepBlock(stage2, prefix: "encoder.down3", stride: 2, identity: false)
        for index in 0..<6
        {
            stage3 = try self.qaRepBlock(stage3, prefix: "encoder.stage3.\(index)", identity: true)
        }
        stage3 = try self.channelAttention(stage3, prefix: "encoder.stage3.6")
        stage3 = try self.globalContext(stage3, prefix: "encoder.stage3.7")

        var stage4 = try self.qaRepBlock(stage3, prefix: "encoder.down4", stride: 2, identity: false)
        for index in 0..<2
        {
            stage4 = try self.qaRepBlock(stage4, prefix: "encoder.stage4.\(index)", identity: true)
        }
        stage4 = try self.sppf(stage4, prefix: "encoder.spp")

        // The cross-scale exchange is scaled by 0.3 after a linear resize or
        // pool; the scale commutes with both, so it is folded into the conv
        // weights instead of running as its own multiply.
        let stage3BeforeCrossScale = stage3
        let lowToHigh = try self.convolution(
            stage4,
            weights: self.scaled(try self.weights.floats(named: "encoder.cross_scale.low_to_high.weight"), by: 0.3),
            shape: try self.weights.shape(named: "encoder.cross_scale.low_to_high.weight"),
            bias: nil,
            groups: 4
        )
        let lowUpsampled = self.resizeNearest(
            lowToHigh,
            height: self.inputHeight / 16,
            width: self.inputWidth / 16
        )
        stage3 = self.graph.addition(stage3, lowUpsampled, name: nil)

        let highToLow = try self.convolution(
            stage3BeforeCrossScale,
            weights: self.scaled(try self.weights.floats(named: "encoder.cross_scale.high_to_low.weight"), by: 0.3),
            shape: try self.weights.shape(named: "encoder.cross_scale.high_to_low.weight"),
            bias: nil,
            groups: 4
        )
        let highDownsampled = try self.averagePool(highToLow, kernel: 2, stride: 2)
        stage4 = self.graph.addition(stage4, highDownsampled, name: nil)

        let decoder4 = try self.convBN(stage4, prefix: "decoder.proj4")
        let decoder3 = try self.fusion(
            high: stage3, low: decoder4, prefix: "decoder.fuse3",
            height: self.inputHeight / 16, width: self.inputWidth / 16
        )
        let decoder2 = try self.fusion(
            high: stage2, low: decoder3, prefix: "decoder.fuse2",
            height: self.inputHeight / 8, width: self.inputWidth / 8
        )
        let decoder1 = try self.fusion(
            high: stage1, low: decoder2, prefix: "decoder.fuse1",
            height: self.inputHeight / 4, width: self.inputWidth / 4
        )
        let decoderHalf = try self.fusion(
            high: stemHalf, low: decoder1, prefix: "decoder.fuse_half",
            height: self.inputHeight / 2, width: self.inputWidth / 2
        )

        let depthHalf = try self.convolution(
            decoderHalf,
            weight: "decoder.head_half.weight",
            bias: "decoder.head_half.bias",
            padding: 1
        )
        return try self.npuUpsample(feature: decoderHalf, depth: depthHalf)
    }

    /// QARep block reparameterized for inference: the 3x3 conv + BN, the
    /// 1x1 conv + BN, and the optional identity branch are all linear in the
    /// input, so they sum exactly into one 3x3 conv with bias. The 1x1 kernel
    /// lands on the 3x3 center (same stride; 3x3 uses padding 1, 1x1 none,
    /// so both read the same input pixel), and the identity adds 1 at the
    /// center of each channel's own kernel.
    private func qaRepBlock(
        _ input: MPSGraphTensor,
        prefix: String,
        stride: Int = 1,
        identity: Bool
    ) throws -> MPSGraphTensor
    {
        let shape3 = try self.weights.shape(named: "\(prefix).branch_3x3.0.weight")
        let shape1 = try self.weights.shape(named: "\(prefix).branch_1x1.0.weight")
        let outputChannels = shape3[0]
        let inputChannels = shape3[1]
        guard shape3 == [outputChannels, inputChannels, 3, 3], shape1 == [outputChannels, inputChannels, 1, 1] else
        {
            throw ZipDepthError("QARep block '\(prefix)' has unexpected branch shapes \(shape3) and \(shape1).")
        }
        guard !identity || outputChannels == inputChannels else
        {
            throw ZipDepthError("QARep block '\(prefix)' has an identity branch but \(inputChannels) -> \(outputChannels) channels.")
        }
        let branch3 = try self.foldBatchNormalization(
            weights: try self.weights.floats(named: "\(prefix).branch_3x3.0.weight"),
            shape: shape3,
            bias: nil,
            prefix: "\(prefix).branch_3x3.1"
        )
        let branch1 = try self.foldBatchNormalization(
            weights: try self.weights.floats(named: "\(prefix).branch_1x1.0.weight"),
            shape: shape1,
            bias: nil,
            prefix: "\(prefix).branch_1x1.1"
        )
        var kernel = branch3.weights
        for outputChannel in 0..<outputChannels
        {
            for inputChannel in 0..<inputChannels
            {
                let center = ((outputChannel * inputChannels + inputChannel) * 3 + 1) * 3 + 1
                kernel[center] += branch1.weights[outputChannel * inputChannels + inputChannel]
                if identity, inputChannel == outputChannel
                {
                    kernel[center] += 1
                }
            }
        }
        let bias = zip(branch3.bias, branch1.bias).map { $0 + $1 }
        let output = try self.convolution(input, weights: kernel, shape: shape3, bias: bias, stride: stride, padding: 1)
        return self.graph.reLU(with: output, name: nil)
    }

    /// `input + BN(depthwise3x3(input) + depthwise3x3_dilation2(input))` as
    /// one depthwise 5x5 conv: the plain 3x3 taps sit at offsets -1...1 and
    /// the dilated taps at -2, 0, 2 inside the 5x5 window (padding 2 covers
    /// both exactly), BN folds per channel, and the residual adds 1 at the
    /// center.
    private func minimalMultiScale(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let shape = try self.weights.shape(named: "\(prefix).branch1.weight")
        let channels = shape[0]
        guard shape == [channels, 1, 3, 3],
              try self.weights.shape(named: "\(prefix).branch2.weight") == shape else
        {
            throw ZipDepthError("Multi-scale block '\(prefix)' has unexpected depthwise shapes.")
        }
        let branch1 = try self.weights.floats(named: "\(prefix).branch1.weight")
        let branch2 = try self.weights.floats(named: "\(prefix).branch2.weight")
        var kernel = [Float](repeating: 0, count: channels * 25)
        for channel in 0..<channels
        {
            for row in 0..<3
            {
                for column in 0..<3
                {
                    let tap = branch1[(channel * 3 + row) * 3 + column]
                    let dilatedTap = branch2[(channel * 3 + row) * 3 + column]
                    kernel[(channel * 5 + row + 1) * 5 + column + 1] += tap
                    kernel[(channel * 5 + row * 2) * 5 + column * 2] += dilatedTap
                }
            }
        }
        let folded = try self.foldBatchNormalization(weights: kernel, shape: [channels, 1, 5, 5], bias: nil, prefix: "\(prefix).bn")
        var fused = folded.weights
        for channel in 0..<channels
        {
            fused[(channel * 5 + 2) * 5 + 2] += 1
        }
        return try self.convolution(input, weights: fused, shape: [channels, 1, 5, 5], bias: folded.bias, groups: channels, padding: 2)
    }

    private func stripPoolingAttention(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let horizontal = self.graph.mean(of: input, axes: [3], name: nil)
        let vertical = self.graph.mean(of: input, axes: [2], name: nil)
        let strips = self.graph.addition(horizontal, vertical, name: nil)
        let normalized = try self.convolutionBatchNormalized(strips, weight: "\(prefix).gate_conv.0.weight", batchNormalization: "\(prefix).gate_conv.1", groups: 96)
        return self.graph.multiplication(input, self.graph.sigmoid(with: normalized, name: nil), name: nil)
    }

    private func channelAttention(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let pooled = self.graph.mean(of: input, axes: [2, 3], name: nil)
        let reduced = self.graph.reLU(
            with: try self.convolution(pooled, weight: "\(prefix).fc.0.weight"),
            name: nil
        )
        let expanded = try self.convolution(reduced, weight: "\(prefix).fc.2.weight")
        return self.graph.multiplication(input, self.graph.sigmoid(with: expanded, name: nil), name: nil)
    }

    private func globalContext(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let height = self.inputHeight / 16
        let width = self.inputWidth / 16
        let spatialCount = height * width
        let contextLogits = try self.convolution(
            input,
            weight: "\(prefix).context_weight.weight",
            bias: "\(prefix).context_weight.bias"
        )
        let flatLogits = self.graph.reshape(contextLogits, shape: [1, 1, NSNumber(value: spatialCount)], name: nil)
        let mask = self.graph.softMax(with: flatLogits, axis: 2, name: nil)
        let flatInput = self.graph.reshape(input, shape: [1, 192, NSNumber(value: spatialCount)], name: nil)
        let transposedMask = self.graph.transpose(mask, permutation: [0, 2, 1], name: nil)
        var context = self.graph.matrixMultiplication(primary: flatInput, secondary: transposedMask, name: nil)
        context = self.graph.reshape(context, shape: [1, 192, 1, 1], name: nil)
        context = self.graph.reLU(with: try self.convolutionBatchNormalized(
            context,
            weight: "\(prefix).transform.0.weight",
            bias: "\(prefix).transform.0.bias",
            batchNormalization: "\(prefix).transform.1"
        ), name: nil)
        context = try self.convolution(
            context,
            weight: "\(prefix).transform.3.weight",
            bias: "\(prefix).transform.3.bias"
        )
        return self.graph.addition(input, context, name: nil)
    }

    private func sppf(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let reduced = try self.convBN(input, prefix: "\(prefix).cv1")
        let pooled1 = try self.maxPool(reduced, kernel: 5, stride: 1, padding: 2)
        let pooled2 = try self.maxPool(pooled1, kernel: 5, stride: 1, padding: 2)
        let pooled3 = try self.maxPool(pooled2, kernel: 5, stride: 1, padding: 2)
        let concatenated = self.graph.concatTensors([reduced, pooled1, pooled2, pooled3], dimension: 1, name: nil)
        return try self.convBN(concatenated, prefix: "\(prefix).cv2")
    }

    private func fusion(
        high: MPSGraphTensor,
        low: MPSGraphTensor,
        prefix: String,
        height: Int,
        width: Int
    ) throws -> MPSGraphTensor
    {
        // BN(highConv + lowConv) == highConv' + lowConv' + shift: the per-
        // channel BN scale folds into both convs, the shift into one bias.
        let highShape = try self.weights.shape(named: "\(prefix).proj_high.weight")
        let lowShape = try self.weights.shape(named: "\(prefix).proj_low.weight")
        let foldedHigh = try self.foldBatchNormalization(
            weights: try self.weights.floats(named: "\(prefix).proj_high.weight"),
            shape: highShape,
            bias: nil,
            prefix: "\(prefix).bn"
        )
        let foldedLow = try self.foldBatchNormalization(
            weights: try self.weights.floats(named: "\(prefix).proj_low.weight"),
            shape: lowShape,
            bias: nil,
            prefix: "\(prefix).bn"
        )
        let projectedHigh = try self.convolution(high, weights: foldedHigh.weights, shape: highShape, bias: foldedHigh.bias, groups: 4)
        let resizedLow = self.resizeBilinear(low, height: height, width: width)
        let projectedLow = try self.convolution(resizedLow, weights: foldedLow.weights, shape: lowShape, bias: nil, groups: 4)
        return self.graph.reLU(with: self.graph.addition(projectedHigh, projectedLow, name: nil), name: nil)
    }

    private func npuUpsample(feature: MPSGraphTensor, depth: MPSGraphTensor) throws -> MPSGraphTensor
    {
        let prefix = "decoder.convex_up.where_conv"
        var alpha = self.graph.reLU(with: try self.convolutionBatchNormalized(
            feature,
            weight: "\(prefix).0.weight",
            batchNormalization: "\(prefix).1"
        ), name: nil)
        alpha = self.graph.reLU(with: try self.convolutionBatchNormalized(
            alpha,
            weight: "\(prefix).3.weight",
            batchNormalization: "\(prefix).4",
            groups: 16,
            padding: 2
        ), name: nil)
        alpha = try self.convolution(alpha, weight: "\(prefix).6.weight")

        alpha = self.graph.sigmoid(
            with: self.resizeBilinear(alpha, height: self.inputHeight, width: self.inputWidth),
            name: nil
        )
        let nearestDepth = self.resizeNearest(depth, height: self.inputHeight, width: self.inputWidth)
        let bilinearDepth = self.resizeBilinear(depth, height: self.inputHeight, width: self.inputWidth)
        let oneMinusAlpha = self.graph.subtraction(self.scalar(1), alpha, name: nil)
        let blended = self.graph.addition(
            self.graph.multiplication(alpha, nearestDepth, name: nil),
            self.graph.multiplication(oneMinusAlpha, bilinearDepth, name: nil),
            name: nil
        )
        let depth = self.graph.reLU(with: blended, name: nil)
        return self.dataType == .float32 ? depth : self.graph.cast(depth, to: .float32, name: "depth")
    }

    private func convBN(
        _ input: MPSGraphTensor,
        prefix: String,
        stride: Int = 1
    ) throws -> MPSGraphTensor
    {
        let convolution = try self.convolutionBatchNormalized(
            input,
            weight: "\(prefix).conv.weight",
            batchNormalization: "\(prefix).bn",
            stride: stride,
            padding: try self.weights.shape(named: "\(prefix).conv.weight")[2] / 2
        )
        return self.graph.reLU(with: convolution, name: nil)
    }

    private func convolution(
        _ input: MPSGraphTensor,
        weight weightName: String,
        bias biasName: String? = nil,
        stride: Int = 1,
        groups: Int = 1,
        dilation: Int = 1,
        padding: Int = 0
    ) throws -> MPSGraphTensor
    {
        try self.convolution(
            input,
            weights: try self.weights.floats(named: weightName),
            shape: try self.weights.shape(named: weightName),
            bias: try biasName.map { try self.weights.floats(named: $0) },
            stride: stride,
            groups: groups,
            dilation: dilation,
            padding: padding
        )
    }

    /// A convolution followed by an inference batch norm, folded into one
    /// convolution with bias.
    private func convolutionBatchNormalized(
        _ input: MPSGraphTensor,
        weight weightName: String,
        bias biasName: String? = nil,
        batchNormalization batchNormalizationPrefix: String,
        stride: Int = 1,
        groups: Int = 1,
        padding: Int = 0
    ) throws -> MPSGraphTensor
    {
        let shape = try self.weights.shape(named: weightName)
        let folded = try self.foldBatchNormalization(
            weights: try self.weights.floats(named: weightName),
            shape: shape,
            bias: try biasName.map { try self.weights.floats(named: $0) },
            prefix: batchNormalizationPrefix
        )
        return try self.convolution(input, weights: folded.weights, shape: shape, bias: folded.bias, stride: stride, groups: groups, padding: padding)
    }

    /// OIHW `weights` with `shape`, optional per-output-channel `bias`.
    private func convolution(
        _ input: MPSGraphTensor,
        weights: [Float],
        shape: [Int],
        bias: [Float]?,
        stride: Int = 1,
        groups: Int = 1,
        dilation: Int = 1,
        padding: Int = 0
    ) throws -> MPSGraphTensor
    {
        guard let descriptor = MPSGraphConvolution2DOpDescriptor(
            strideInX: stride,
            strideInY: stride,
            dilationRateInX: dilation,
            dilationRateInY: dilation,
            groups: groups,
            paddingLeft: padding,
            paddingRight: padding,
            paddingTop: padding,
            paddingBottom: padding,
            paddingStyle: .explicit,
            dataLayout: .NCHW,
            weightsLayout: .OIHW
        ) else
        {
            throw ZipDepthError("Could not create a convolution descriptor for shape \(shape).")
        }
        // mixedFloat16 casts into and out of each convolution; float32 and
        // float16 already match and add no casts.
        let weightTensor = self.constant(weights, shape: shape.map(NSNumber.init(value:)), dataType: self.layerDataType)
        let layerInput = input.dataType == self.layerDataType ? input : self.graph.cast(input, to: self.layerDataType, name: nil)
        var output = self.graph.convolution2D(layerInput, weights: weightTensor, descriptor: descriptor, name: nil)
        if let bias
        {
            output = self.graph.addition(output, self.constant(bias, shape: [1, NSNumber(value: bias.count), 1, 1], dataType: self.layerDataType), name: nil)
        }
        return output.dataType == self.dataType ? output : self.graph.cast(output, to: self.dataType, name: nil)
    }

    /// Folds the inference batch norm at `prefix` into OIHW conv `weights`:
    /// with s = gamma / sqrt(var + eps), W'[o] = W[o] * s[o] and
    /// b'[o] = b[o] * s[o] + (beta[o] - mean[o] * s[o]), where b is the
    /// conv's own bias (zero when it has none).
    private func foldBatchNormalization(
        weights: [Float],
        shape: [Int],
        bias: [Float]?,
        prefix: String
    ) throws -> (weights: [Float], bias: [Float])
    {
        let gamma = try self.weights.floats(named: "\(prefix).weight")
        let beta = try self.weights.floats(named: "\(prefix).bias")
        let mean = try self.weights.floats(named: "\(prefix).running_mean")
        let variance = try self.weights.floats(named: "\(prefix).running_var")
        let outputChannels = shape[0]
        guard gamma.count == outputChannels, beta.count == outputChannels,
              mean.count == outputChannels, variance.count == outputChannels,
              bias == nil || bias?.count == outputChannels else
        {
            throw ZipDepthError("Batch norm '\(prefix)' does not match a \(outputChannels)-channel convolution.")
        }
        let valuesPerOutputChannel = shape.dropFirst().reduce(1, *)
        var folded = weights
        var foldedBias = [Float](repeating: 0, count: outputChannels)
        for outputChannel in 0..<outputChannels
        {
            let scale = gamma[outputChannel] / sqrt(variance[outputChannel] + 0.00001)
            let start = outputChannel * valuesPerOutputChannel
            for index in start..<(start + valuesPerOutputChannel)
            {
                folded[index] *= scale
            }
            foldedBias[outputChannel] = (bias?[outputChannel] ?? 0) * scale + beta[outputChannel] - mean[outputChannel] * scale
        }
        return (folded, foldedBias)
    }

    private func scaled(_ values: [Float], by factor: Float) -> [Float]
    {
        values.map { $0 * factor }
    }

    private func resizeBilinear(_ input: MPSGraphTensor, height: Int, width: Int) -> MPSGraphTensor
    {
        self.graph.resize(
            input,
            size: [NSNumber(value: height), NSNumber(value: width)],
            mode: .bilinear,
            centerResult: true,
            alignCorners: false,
            layout: .NCHW,
            name: nil
        )
    }

    private func resizeNearest(_ input: MPSGraphTensor, height: Int, width: Int) -> MPSGraphTensor
    {
        self.graph.resize(
            input,
            size: [NSNumber(value: height), NSNumber(value: width)],
            mode: .nearest,
            centerResult: false,
            alignCorners: false,
            layout: .NCHW,
            name: nil
        )
    }

    private func maxPool(_ input: MPSGraphTensor, kernel: Int, stride: Int, padding: Int) throws -> MPSGraphTensor
    {
        guard let descriptor = MPSGraphPooling2DOpDescriptor(
            kernelWidth: kernel,
            kernelHeight: kernel,
            strideInX: stride,
            strideInY: stride,
            dilationRateInX: 1,
            dilationRateInY: 1,
            paddingLeft: padding,
            paddingRight: padding,
            paddingTop: padding,
            paddingBottom: padding,
            paddingStyle: .explicit,
            dataLayout: .NCHW
        ) else
        {
            throw ZipDepthError("Could not create a max-pooling descriptor.")
        }
        return self.graph.maxPooling2D(withSourceTensor: input, descriptor: descriptor, name: nil)
    }

    private func averagePool(_ input: MPSGraphTensor, kernel: Int, stride: Int) throws -> MPSGraphTensor
    {
        guard let descriptor = MPSGraphPooling2DOpDescriptor(
            kernelWidth: kernel,
            kernelHeight: kernel,
            strideInX: stride,
            strideInY: stride,
            dilationRateInX: 1,
            dilationRateInY: 1,
            paddingLeft: 0,
            paddingRight: 0,
            paddingTop: 0,
            paddingBottom: 0,
            paddingStyle: .explicit,
            dataLayout: .NCHW
        ) else
        {
            throw ZipDepthError("Could not create an average-pooling descriptor.")
        }
        descriptor.includeZeroPadToAverage = false
        return self.graph.avgPooling2D(withSourceTensor: input, descriptor: descriptor, name: nil)
    }

    private func broadcastConstant(named name: String) throws -> MPSGraphTensor
    {
        let values = try self.weights.floats(named: name)
        return self.channelConstant(values)
    }

    private func channelConstant(_ values: [Float]) -> MPSGraphTensor
    {
        self.constant(values, shape: [1, NSNumber(value: values.count), 1, 1])
    }

    private func scalar(_ value: Float) -> MPSGraphTensor
    {
        self.constant([value], shape: [1])
    }

    /// A constant in `dataType` (the graph's activation type by default).
    /// Built as float32 and cast in the graph when needed -- MPSGraph folds a
    /// cast of a constant at compile time -- rather than with Swift's
    /// `Float16`, which Intel Macs lack.
    private func constant(_ values: [Float], shape: [NSNumber], dataType: MPSDataType? = nil) -> MPSGraphTensor
    {
        let targetType = dataType ?? self.dataType
        let float32Constant = self.graph.constant(Data(bytes: values, count: values.count * MemoryLayout<Float>.stride), shape: shape, dataType: .float32)
        return targetType == .float32 ? float32Constant : self.graph.cast(float32Constant, to: targetType, name: nil)
    }
}
