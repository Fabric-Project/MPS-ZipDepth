import Foundation
import Metal
import MetalPerformanceShaders
import Testing
@testable import MPSZipDepth

@Test func rejectsInvalidInputDimensions() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }

    #expect(throws: ZipDepthError.self) {
        try ZipDepthMPSGraph(inputWidth: 383, inputHeight: 384, commandQueue: commandQueue)
    }
}

@Test func rejectsWrongInputArraySize() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }

    let model = try ZipDepthMPSGraph(inputWidth: 32, inputHeight: 32, commandQueue: commandQueue)
    #expect(throws: ZipDepthError.self) {
        try model.prediction(rgb: [])
    }
}

@Test func runsOfficialGraphEndToEnd() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }

    let model = try ZipDepthMPSGraph(inputWidth: 384, inputHeight: 384, commandQueue: commandQueue)
    let rgb = [Float](repeating: 0.5, count: 384 * 384 * 3)
    guard let inputBuffer = device.makeBuffer(
        bytes: rgb,
        length: rgb.count * MemoryLayout<Float>.stride
    ) else
    {
        return
    }
    let depth = try model.run(inputBuffer: inputBuffer)

    #expect(depth.count == 384 * 384)
    #expect(depth.allSatisfy { $0.isFinite })
    #expect(depth.allSatisfy { $0 >= 0 })
    let meanDepth = depth.reduce(0, +) / Float(depth.count)
    #expect(abs(meanDepth - 0.046175327) < 0.0005)
}

@Test func submitsOfficialGraphAsynchronously() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let commandBuffer = MPSCommandBuffer(from: commandQueue)

    let model = try ZipDepthMPSGraph(
        inputWidth: 32,
        inputHeight: 32,
        commandQueue: commandQueue,
        maxFramesInFlight: 1
    )
    let rgb = [Float](repeating: 0.5, count: 32 * 32 * 3)
    guard let inputBuffer = device.makeBuffer(
        bytes: rgb,
        length: rgb.count * MemoryLayout<Float>.stride
    ) else
    {
        return
    }

    let completionSemaphore = DispatchSemaphore(value: 0)
    let resultLock = NSLock()
    var submittedResult: Result<[Float], any Error>?
    let wasSubmitted = try model.submit(
        inputBuffer: inputBuffer,
        commandBuffer: commandBuffer
    ) { result in
        resultLock.lock()
        submittedResult = result
        resultLock.unlock()
        completionSemaphore.signal()
    }
    commandBuffer.commit()

    #expect(wasSubmitted)
    #expect(completionSemaphore.wait(timeout: .now() + 10) == .success)
    resultLock.lock()
    let capturedResult = submittedResult
    resultLock.unlock()
    let depth = try capturedResult?.get()
    #expect(depth?.count == 32 * 32)
}

/// Whether a second command buffer on the *same* queue can safely read
/// `encode()`'s output with no explicit CPU-side wait in between -- i.e.
/// whether plain same-queue commit-order + Metal's automatic hazard tracking
/// is sufficient for `MPSGraphExecutable`, the way it already is for plain
/// `MPSKernel`/`MPSUnaryImageKernel` encodes (see Satin's
/// `ARDepthUpscaler`). This is NOT assumed -- `MPSGraphExecutable` is opaque
/// (may route through the ANE, does its own internal `MPSCommandBuffer`
/// continuation) in a way plain compute kernels are not, so the guarantee
/// does not automatically transfer by analogy. If this test is flaky or
/// fails, same-queue ordering alone is not enough and some explicit
/// completion signal is still required after `encode()`.
@Test func encodeOutputIsReadableFromASecondCommandBufferWithNoInterveningWait() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }

    let model = try ZipDepthMPSGraph(inputWidth: 384, inputHeight: 384, commandQueue: commandQueue)
    let rgb = [Float](repeating: 0.5, count: 384 * 384 * 3)
    guard let inputBuffer = device.makeBuffer(
        bytes: rgb,
        length: rgb.count * MemoryLayout<Float>.stride
    ),
          let outputBuffer = device.makeBuffer(length: model.outputBufferLength, options: .storageModePrivate),
          let verifyBuffer = device.makeBuffer(length: model.outputBufferLength, options: .storageModeShared),
          let verifyCommandBuffer = commandQueue.makeCommandBuffer() else
    {
        return
    }
    let encodeCommandBuffer = MPSCommandBuffer(from: commandQueue)

    let wasEncoded = try model.encode(
        inputBuffer: inputBuffer,
        outputBuffer: outputBuffer,
        commandBuffer: encodeCommandBuffer
    )
    #expect(wasEncoded)
    encodeCommandBuffer.commit()
    // Deliberately no wait here -- this is the exact gap under test.

    guard let blitEncoder = verifyCommandBuffer.makeBlitCommandEncoder() else
    {
        return
    }
    blitEncoder.copy(
        from: outputBuffer, sourceOffset: 0,
        to: verifyBuffer, destinationOffset: 0,
        size: model.outputBufferLength
    )
    blitEncoder.endEncoding()
    verifyCommandBuffer.commit()
    verifyCommandBuffer.waitUntilCompleted()

    #expect(verifyCommandBuffer.status == .completed)
    #expect(verifyCommandBuffer.error == nil)

    let count = 384 * 384
    let depth = Array(UnsafeBufferPointer(
        start: verifyBuffer.contents().assumingMemoryBound(to: Float.self),
        count: count
    ))
    #expect(depth.allSatisfy { $0.isFinite })
    #expect(depth.allSatisfy { $0 >= 0 })
    let meanDepth = depth.reduce(0, +) / Float(depth.count)
    #expect(abs(meanDepth - 0.046175327) < 0.0005)
}

/// `encode()`'s slot protection drops an overlapping call rather than
/// allowing two executions to race through the graph's shared intermediate
/// storage. Once the first call completes and releases its slot, retrying
/// the second call succeeds.
@Test func encodeDropsAnOverlappingCallUntilItsSlotIsReleased() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }

    let model = try ZipDepthMPSGraph(
        inputWidth: 384,
        inputHeight: 384,
        commandQueue: commandQueue,
        maxFramesInFlight: 1
    )
    let rgb = [Float](repeating: 0.5, count: 384 * 384 * 3)
    guard let inputBufferA = device.makeBuffer(bytes: rgb, length: rgb.count * MemoryLayout<Float>.stride),
          let inputBufferB = device.makeBuffer(bytes: rgb, length: rgb.count * MemoryLayout<Float>.stride),
          let outputBufferA = device.makeBuffer(length: model.outputBufferLength, options: .storageModePrivate),
          let outputBufferB = device.makeBuffer(length: model.outputBufferLength, options: .storageModePrivate),
          let verifyBufferA = device.makeBuffer(length: model.outputBufferLength, options: .storageModeShared),
          let verifyBufferB = device.makeBuffer(length: model.outputBufferLength, options: .storageModeShared) else
    {
        return
    }

    let commandBufferA = MPSCommandBuffer(from: commandQueue)
    let commandBufferB = MPSCommandBuffer(from: commandQueue)
    let firstWasEncoded = try model.encode(
        inputBuffer: inputBufferA,
        outputBuffer: outputBufferA,
        commandBuffer: commandBufferA
    )
    let overlappingCallWasEncoded = try model.encode(
        inputBuffer: inputBufferB,
        outputBuffer: outputBufferB,
        commandBuffer: commandBufferB
    )
    #expect(firstWasEncoded)
    #expect(!overlappingCallWasEncoded)

    commandBufferA.commit()
    commandBufferA.waitUntilCompleted()

    let retryCommandBuffer = MPSCommandBuffer(from: commandQueue)
    let retryWasEncoded = try model.encode(
        inputBuffer: inputBufferB,
        outputBuffer: outputBufferB,
        commandBuffer: retryCommandBuffer
    )
    #expect(retryWasEncoded)
    retryCommandBuffer.commit()

    guard let verifyCommandBuffer = commandQueue.makeCommandBuffer(),
          let blitEncoder = verifyCommandBuffer.makeBlitCommandEncoder() else
    {
        return
    }
    blitEncoder.copy(from: outputBufferA, sourceOffset: 0, to: verifyBufferA, destinationOffset: 0, size: model.outputBufferLength)
    blitEncoder.copy(from: outputBufferB, sourceOffset: 0, to: verifyBufferB, destinationOffset: 0, size: model.outputBufferLength)
    blitEncoder.endEncoding()
    verifyCommandBuffer.commit()
    verifyCommandBuffer.waitUntilCompleted()

    #expect(verifyCommandBuffer.status == .completed)
    #expect(verifyCommandBuffer.error == nil)

    let count = 384 * 384
    for verifyBuffer in [verifyBufferA, verifyBufferB]
    {
        let depth = Array(UnsafeBufferPointer(
            start: verifyBuffer.contents().assumingMemoryBound(to: Float.self),
            count: count
        ))
        #expect(depth.allSatisfy { $0.isFinite })
        #expect(depth.allSatisfy { $0 >= 0 })
        let meanDepth = depth.reduce(0, +) / Float(depth.count)
        #expect(abs(meanDepth - 0.046175327) < 0.0005)
    }
}

/// `encode()` never commits: the caller keeps encoding its own work onto the
/// same `MPSCommandBuffer` after the model and commits it once. This guards
/// the caller-owned command-buffer contract -- if the package committed
/// internally, the caller's later encode or commit would hit an
/// already-committed buffer.
@Test func encodeLeavesTheCommandBufferForTheCallerToCommit() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }

    let model = try ZipDepthMPSGraph(inputWidth: 32, inputHeight: 32, commandQueue: commandQueue)

    guard let inputBuffer = device.makeBuffer(length: model.inputBufferLength, options: .storageModePrivate),
          let outputBuffer = device.makeBuffer(length: model.outputBufferLength, options: .storageModePrivate),
          let verifyBuffer = device.makeBuffer(length: model.outputBufferLength, options: .storageModeShared) else
    {
        return
    }

    let commandBuffer = MPSCommandBuffer(from: commandQueue)
    let wasEncoded = try model.encode(inputBuffer: inputBuffer, outputBuffer: outputBuffer, commandBuffer: commandBuffer)
    #expect(wasEncoded)

    // Caller-owned follow-up work on the same buffer, then one commit.
    guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else
    {
        return
    }
    blitEncoder.copy(from: outputBuffer, sourceOffset: 0, to: verifyBuffer, destinationOffset: 0, size: model.outputBufferLength)
    blitEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    #expect(commandBuffer.status == .completed)
    #expect(commandBuffer.error == nil)
    let depth = Array(UnsafeBufferPointer(
        start: verifyBuffer.contents().assumingMemoryBound(to: Float.self),
        count: 32 * 32
    ))
    #expect(depth.allSatisfy { $0.isFinite })
}

/// Full-output regression gate for graph rewrites (batch-norm folding,
/// QARep reparameterization, and similar exact transforms). Runs a
/// deterministic patterned 384x384 image and compares every depth value
/// against `Fixtures/reference_depth_384.bin`, recorded from the unfolded
/// graph. Record or re-record it deliberately with
/// `ZIPDEPTH_WRITE_REFERENCE=1 swift test --filter foldedGraphMatchesReference`.
/// `ZIPDEPTH_PRECISION=float16` compares the float16 graph against the same
/// float32 reference, gated at the float16 bar instead of the exact one.
@Test func foldedGraphMatchesReference() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }

    let width = 384
    let height = 384
    var rgb = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height
    {
        for x in 0..<width
        {
            let index = (y * width + x) * 3
            let fx = Float(x) / Float(width)
            let fy = Float(y) / Float(height)
            rgb[index + 0] = 0.5 + 0.5 * sin(fx * 17 + fy * 3)
            rgb[index + 1] = 0.5 + 0.5 * cos(fy * 23 - fx * 5)
            rgb[index + 2] = fx * fy
        }
    }
    let inputBuffer = try #require(device.makeBuffer(bytes: rgb, length: rgb.count * MemoryLayout<Float>.stride))
    let precision = zipDepthTestPrecision()
    let model = try ZipDepthMPSGraph(inputWidth: width, inputHeight: height, commandQueue: commandQueue, precision: precision)
    let depth = try model.run(inputBuffer: inputBuffer)

    let fixtureURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appending(path: "Fixtures/reference_depth_384.bin")
    if ProcessInfo.processInfo.environment["ZIPDEPTH_WRITE_REFERENCE"] != nil
    {
        try #require(precision == .float32, "record the reference from the float32 graph")
        try depth.withUnsafeBytes { try Data($0).write(to: fixtureURL) }
        print("Wrote ZipDepth reference output to \(fixtureURL.path)")
        return
    }

    let referenceData = try Data(contentsOf: fixtureURL)
    let reference = referenceData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    try #require(reference.count == depth.count)

    var maximumError: Float = 0
    var squaredErrorSum: Double = 0
    var absoluteErrorSum: Double = 0
    for (actual, expected) in zip(depth, reference)
    {
        let error = abs(actual - expected)
        maximumError = max(maximumError, error)
        absoluteErrorSum += Double(error)
        squaredErrorSum += Double(error) * Double(error)
    }
    let peak = Double(reference.max() ?? 1)
    let meanSquaredError = squaredErrorSum / Double(reference.count)
    let psnr = meanSquaredError == 0 ? Double.infinity : 10 * log10(peak * peak / meanSquaredError)
    print("ZipDepth \(precision) vs reference: MAE=\(absoluteErrorSum / Double(reference.count)) max=\(maximumError) PSNR=\(psnr) dB")
    // Exact float32 rewrites stay above 70 dB; float16 compute is held to
    // the Core AI float16 bar (investigate below 40 dB).
    #expect(psnr > (precision == .float32 ? 70 : 40))
}

/// Opt-in steady-state benchmark: graph construction time, then per-frame CPU
/// encode cost and queued wall time over 60 frames submitted without waits.
/// `ZIPDEPTH_RUN_BENCHMARK=1 swift test --filter zipDepthSteadyStatePerformance`
/// (`ZIPDEPTH_BENCHMARK_SIZE` overrides the default 384).
@Test func zipDepthSteadyStatePerformance() throws
{
    guard ProcessInfo.processInfo.environment["ZIPDEPTH_RUN_BENCHMARK"] != nil,
          let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }
    let size = Int(ProcessInfo.processInfo.environment["ZIPDEPTH_BENCHMARK_SIZE"] ?? "") ?? 384
    let clock = ContinuousClock()
    func milliseconds(_ duration: Duration) -> Double
    {
        Double(duration.components.seconds) * 1_000 + Double(duration.components.attoseconds) / 1e15
    }

    let constructionStart = clock.now
    let precision = zipDepthTestPrecision()
    let model = try ZipDepthMPSGraph(inputWidth: size, inputHeight: size, commandQueue: commandQueue, maxFramesInFlight: 16, precision: precision)
    let construction = clock.now - constructionStart

    let inputBuffer = try #require(device.makeBuffer(length: model.inputBufferLength, options: .storageModePrivate))
    let outputBuffer = try #require(device.makeBuffer(length: model.outputBufferLength, options: .storageModePrivate))
    func encodeFrame() throws -> (MPSCommandBuffer, Double)
    {
        let rawCommandBuffer = try #require(commandQueue.makeCommandBuffer())
        let commandBuffer = MPSCommandBuffer(commandBuffer: rawCommandBuffer)
        let start = clock.now
        let accepted = try model.encode(inputBuffer: inputBuffer, outputBuffer: outputBuffer, commandBuffer: commandBuffer)
        commandBuffer.commit()
        #expect(accepted)
        return (commandBuffer, milliseconds(clock.now - start))
    }

    for _ in 0..<10 { try encodeFrame().0.waitUntilCompleted() }

    let measuredStart = clock.now
    var cpuSamples: [Double] = []
    var last: MPSCommandBuffer?
    for _ in 0..<60
    {
        let (commandBuffer, cpu) = try encodeFrame()
        cpuSamples.append(cpu)
        last = commandBuffer
        // Keep at most a few frames queued so the slot pool never drops one.
        if cpuSamples.count % 8 == 0 { commandBuffer.waitUntilCompleted() }
    }
    last?.waitUntilCompleted()
    let measured = clock.now - measuredStart

    print("ZipDepth \(size)x\(size) \(precision) graph construction: \(milliseconds(construction) / 1_000) s")
    print("ZipDepth measured CPU encode: \(cpuSamples.reduce(0, +) / Double(cpuSamples.count)) ms/frame")
    print("ZipDepth measured queued wall time: \(milliseconds(measured) / 60) ms/frame")
}

private func zipDepthTestPrecision() -> ZipDepthPrecision
{
    ProcessInfo.processInfo.environment["ZIPDEPTH_PRECISION"] == "float16" ? .float16 : .float32
}
