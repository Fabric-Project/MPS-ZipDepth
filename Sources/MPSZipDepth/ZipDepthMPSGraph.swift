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
        maxFramesInFlight: Int = 3
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
            inputHeight: inputHeight
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
        descriptor.optimizationLevel = .level1
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

    /// Asynchronous MPS-MediaPipe-style inference entry point. Throws for a
    /// genuinely invalid call (wrong-sized buffer, mismatched device) --
    /// those are programmer errors, not transient state, so they surface
    /// immediately rather than looking identical to ordinary backpressure.
    /// Returns `false` (doesn't throw) only when every `maxFramesInFlight`
    /// slot is already occupied, since that's expected, recoverable
    /// pressure a caller should just retry next frame. Never commits
    /// `commandBuffer` -- that decision belongs to whoever created it, not
    /// to this method. The caller must commit it after this call returns
    /// `true`, or `completion` never fires.
    @discardableResult
    public func submit(
        inputBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer,
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

        let mpsCommandBuffer = MPSCommandBuffer(commandBuffer: commandBuffer)
        _ = self.executable.encode(
            to: mpsCommandBuffer,
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

    /// Encodes inference into an existing Metal command buffer without a CPU
    /// readback. Never commits `commandBuffer` -- that decision belongs
    /// entirely to whoever created it, not to this method.
    /// `MPSGraphExecutable.encode(to:)` doesn't commit anything on its own
    /// either, so `commandBuffer` just holds this graph's encoded work,
    /// in order alongside whatever else the caller encodes onto it, until
    /// its owner commits it. Pass a dedicated buffer this call owns
    /// exclusively (any preceding GPU preprocessing may already be encoded
    /// on it) and commit it immediately afterward yourself.
    ///
    /// No CPU wait is required before reading `outputBuffer` once the caller
    /// has committed `commandBuffer`: encode any further work that depends
    /// on it (postprocessing, etc.) onto a *different* command buffer
    /// committed to this same `MTLCommandQueue` afterward -- Metal orders
    /// command buffers on one queue by commit order and automatically
    /// tracks the dependency on `outputBuffer` across that boundary, so no
    /// explicit synchronization is needed.
    ///
    /// Drops the call (returns `false`, encodes nothing) instead of
    /// blocking if all `maxFramesInFlight` slots are already in flight on
    /// the GPU, matching `submit()` and MPS-MediaPipe's `encode()` -- real
    /// backpressure should mean a dropped frame, not a CPU stall, so this
    /// never blocks the caller. The slot this call does acquire is released
    /// once `commandBuffer` completes, however far in the future that turns
    /// out to be, rather than after a CPU wait; it's the same protection
    /// `run()`/`submit()` use to guard the graph's own internal
    /// intermediate-tensor storage from being reused by an overlapping
    /// in-flight call.
    @discardableResult
    public func encode(
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        commandBuffer: MTLCommandBuffer
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

        // Must be registered before commandBuffer is committed, whenever
        // that ends up happening -- Metal requires completion handlers to
        // be added before commit.
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.releaseSlot(slot)
        }

        // waitUntilCompleted = false avoids an extra implicit GPU wait
        // inside this call itself -- see the doc comment above for why no
        // CPU wait is needed afterward either.
        let executionDescriptor = MPSGraphExecutableExecutionDescriptor()
        executionDescriptor.waitUntilCompleted = false

        let mpsCommandBuffer = MPSCommandBuffer(commandBuffer: commandBuffer)
        _ = self.executable.encode(
            to: mpsCommandBuffer,
            inputs: [inputData],
            results: [outputData],
            executionDescriptor: executionDescriptor
        )
        // No commit here -- see the doc comment above; the caller owns that.
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

    func build(inputNHWC: MPSGraphTensor) throws -> MPSGraphTensor
    {
        var input = self.graph.transpose(inputNHWC, permutation: [0, 3, 1, 2], name: "input_nchw")
        let mean = try self.broadcastConstant(named: "mean")
        let standardDeviation = try self.broadcastConstant(named: "std")
        input = self.graph.division(
            self.graph.subtraction(input, mean, name: nil),
            standardDeviation,
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

        let stage3BeforeCrossScale = stage3
        let lowToHigh = try self.convolution(
            stage4,
            weight: "encoder.cross_scale.low_to_high.weight",
            groups: 4
        )
        let lowUpsampled = self.resizeNearest(
            lowToHigh,
            height: self.inputHeight / 16,
            width: self.inputWidth / 16
        )
        stage3 = self.graph.addition(stage3, self.scale(lowUpsampled, by: 0.3), name: nil)

        let highToLow = try self.convolution(
            stage3BeforeCrossScale,
            weight: "encoder.cross_scale.high_to_low.weight",
            groups: 4
        )
        let highDownsampled = try self.averagePool(highToLow, kernel: 2, stride: 2)
        stage4 = self.graph.addition(stage4, self.scale(highDownsampled, by: 0.3), name: nil)

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

    private func qaRepBlock(
        _ input: MPSGraphTensor,
        prefix: String,
        stride: Int = 1,
        identity: Bool
    ) throws -> MPSGraphTensor
    {
        let branch3 = try self.batchNormalize(
            self.convolution(input, weight: "\(prefix).branch_3x3.0.weight", stride: stride, padding: 1),
            prefix: "\(prefix).branch_3x3.1"
        )
        let branch1 = try self.batchNormalize(
            self.convolution(input, weight: "\(prefix).branch_1x1.0.weight", stride: stride),
            prefix: "\(prefix).branch_1x1.1"
        )
        var output = self.graph.addition(branch3, branch1, name: nil)
        if identity
        {
            output = self.graph.addition(output, input, name: nil)
        }
        return self.graph.reLU(with: output, name: nil)
    }

    private func minimalMultiScale(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let branch1 = try self.convolution(input, weight: "\(prefix).branch1.weight", groups: 96, padding: 1)
        let branch2 = try self.convolution(input, weight: "\(prefix).branch2.weight", groups: 96, dilation: 2, padding: 2)
        let combined = self.graph.addition(branch1, branch2, name: nil)
        return self.graph.addition(input, try self.batchNormalize(combined, prefix: "\(prefix).bn"), name: nil)
    }

    private func stripPoolingAttention(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let horizontal = self.graph.mean(of: input, axes: [3], name: nil)
        let vertical = self.graph.mean(of: input, axes: [2], name: nil)
        let strips = self.graph.addition(horizontal, vertical, name: nil)
        let gateConv = try self.convolution(strips, weight: "\(prefix).gate_conv.0.weight", groups: 96)
        let normalized = try self.batchNormalize(gateConv, prefix: "\(prefix).gate_conv.1")
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
        context = try self.convolution(
            context,
            weight: "\(prefix).transform.0.weight",
            bias: "\(prefix).transform.0.bias"
        )
        context = self.graph.reLU(with: try self.batchNormalize(context, prefix: "\(prefix).transform.1"), name: nil)
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
        let projectedHigh = try self.convolution(high, weight: "\(prefix).proj_high.weight", groups: 4)
        let resizedLow = self.resizeBilinear(low, height: height, width: width)
        let projectedLow = try self.convolution(resizedLow, weight: "\(prefix).proj_low.weight", groups: 4)
        let sum = self.graph.addition(projectedHigh, projectedLow, name: nil)
        return self.graph.reLU(with: try self.batchNormalize(sum, prefix: "\(prefix).bn"), name: nil)
    }

    private func npuUpsample(feature: MPSGraphTensor, depth: MPSGraphTensor) throws -> MPSGraphTensor
    {
        let prefix = "decoder.convex_up.where_conv"
        var alpha = try self.convolution(feature, weight: "\(prefix).0.weight")
        alpha = self.graph.reLU(with: try self.batchNormalize(alpha, prefix: "\(prefix).1"), name: nil)
        alpha = try self.convolution(alpha, weight: "\(prefix).3.weight", groups: 16, padding: 2)
        alpha = self.graph.reLU(with: try self.batchNormalize(alpha, prefix: "\(prefix).4"), name: nil)
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
        return self.graph.reLU(with: blended, name: "depth")
    }

    private func convBN(
        _ input: MPSGraphTensor,
        prefix: String,
        stride: Int = 1
    ) throws -> MPSGraphTensor
    {
        let convolution = try self.convolution(
            input,
            weight: "\(prefix).conv.weight",
            stride: stride,
            padding: try self.weights.shape(named: "\(prefix).conv.weight")[2] / 2
        )
        return self.graph.reLU(with: try self.batchNormalize(convolution, prefix: "\(prefix).bn"), name: nil)
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
            throw ZipDepthError("Could not create the convolution descriptor for '\(weightName)'.")
        }
        var output = self.graph.convolution2D(
            input,
            weights: try self.weights.constant(self.graph, named: weightName),
            descriptor: descriptor,
            name: weightName
        )
        if let biasName
        {
            output = self.graph.addition(output, try self.broadcastConstant(named: biasName), name: nil)
        }
        return output
    }

    private func batchNormalize(_ input: MPSGraphTensor, prefix: String) throws -> MPSGraphTensor
    {
        let gamma = try self.weights.floats(named: "\(prefix).weight")
        let beta = try self.weights.floats(named: "\(prefix).bias")
        let mean = try self.weights.floats(named: "\(prefix).running_mean")
        let variance = try self.weights.floats(named: "\(prefix).running_var")
        let scaleValues = zip(gamma, variance).map { $0 / sqrt($1 + 0.00001) }
        let biasValues = zip(zip(beta, mean), scaleValues).map { pair, scale in pair.0 - pair.1 * scale }
        let scale = self.channelConstant(scaleValues)
        let bias = self.channelConstant(biasValues)
        return self.graph.addition(self.graph.multiplication(input, scale, name: nil), bias, name: nil)
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
        self.graph.constant(
            Data(bytes: values, count: values.count * MemoryLayout<Float>.stride),
            shape: [1, NSNumber(value: values.count), 1, 1],
            dataType: .float32
        )
    }

    private func scalar(_ value: Float) -> MPSGraphTensor
    {
        var value = value
        return withUnsafeBytes(of: &value) { bytes in
            self.graph.constant(Data(bytes), shape: [1], dataType: .float32)
        }
    }

    private func scale(_ input: MPSGraphTensor, by value: Float) -> MPSGraphTensor
    {
        self.graph.multiplication(input, self.scalar(value), name: nil)
    }
}
