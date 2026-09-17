import Foundation
import Metal
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
          let commandQueue = device.makeCommandQueue(),
          let commandBuffer = commandQueue.makeCommandBuffer() else
    {
        return
    }

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
          let encodeCommandBuffer = commandQueue.makeCommandBuffer(),
          let verifyCommandBuffer = commandQueue.makeCommandBuffer() else
    {
        return
    }

    try model.encode(inputBuffer: inputBuffer, outputBuffer: outputBuffer, commandBuffer: encodeCommandBuffer)
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

/// `encode()`'s slot protection should serialize two overlapping in-flight
/// calls against the same model instance rather than let them race and
/// corrupt the graph's shared internal intermediate-tensor storage --
/// `maxFramesInFlight: 1` forces the second call to contend for the only
/// slot, which should make it block in `acquireSlotBlocking()` until the
/// first call's GPU work completes and releases it, not crash or produce
/// corrupted output.
@Test func encodeSerializesOverlappingCallsViaSlotProtection() throws
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
          let verifyBufferB = device.makeBuffer(length: model.outputBufferLength, options: .storageModeShared),
          let commandBufferA = commandQueue.makeCommandBuffer(),
          let commandBufferB = commandQueue.makeCommandBuffer() else
    {
        return
    }

    try model.encode(inputBuffer: inputBufferA, outputBuffer: outputBufferA, commandBuffer: commandBufferA)
    // With maxFramesInFlight: 1, this call has to wait for slot A's release
    // (registered on commandBufferA's completion) before it can proceed --
    // exercising the exact contention path the protection exists for.
    try model.encode(inputBuffer: inputBufferB, outputBuffer: outputBufferB, commandBuffer: commandBufferB)

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

/// By the time `encode(inputBuffer:outputBuffer:commandBuffer:)` returns, the
/// raw `commandBuffer` object passed in is already committed -- callers must
/// treat it as done and encode any further GPU work (postprocessing, etc.)
/// on a fresh command buffer instead. This guards against a regression where
/// a caller mistakenly keeps encoding onto (or calls `.commit()` on) the
/// same reference and crashes on an already-committed command buffer.
@Test func encodeCommitsItsCommandBufferAndOutputIsReadableFromAFreshOne() throws
{
    guard let device = MTLCreateSystemDefaultDevice(),
          let commandQueue = device.makeCommandQueue() else
    {
        return
    }

    let model = try ZipDepthMPSGraph(inputWidth: 32, inputHeight: 32, commandQueue: commandQueue)

    guard let inputBuffer = device.makeBuffer(length: model.inputBufferLength, options: .storageModePrivate),
          let outputBuffer = device.makeBuffer(length: model.outputBufferLength, options: .storageModePrivate),
          let modelCommandBuffer = commandQueue.makeCommandBuffer() else
    {
        return
    }

    // encode() commits `modelCommandBuffer` internally -- calling commit()
    // again here would assert on an already-committed buffer.
    try model.encode(inputBuffer: inputBuffer, outputBuffer: outputBuffer, commandBuffer: modelCommandBuffer)
    modelCommandBuffer.waitUntilCompleted()
    #expect(modelCommandBuffer.status == .completed)
    #expect(modelCommandBuffer.error == nil)

    // A correct caller (mirroring ZipDepthNode) starts a new command buffer
    // for postprocessing rather than reusing `modelCommandBuffer`.
    guard let postprocessCommandBuffer = commandQueue.makeCommandBuffer(),
          let postprocessEncoder = postprocessCommandBuffer.makeComputeCommandEncoder() else
    {
        return
    }
    postprocessEncoder.endEncoding()
    postprocessCommandBuffer.commit()
    postprocessCommandBuffer.waitUntilCompleted()
    #expect(postprocessCommandBuffer.status == .completed)
    #expect(postprocessCommandBuffer.error == nil)
}
