import Foundation
import Metal

extension ZipDepthMPSGraph
{
    public convenience init(
        inputWidth: Int = 384,
        inputHeight: Int = 384,
        commandQueue: MTLCommandQueue,
        maxFramesInFlight: Int = 3
    ) throws
    {
        guard let binaryURL = Bundle.module.url(
            forResource: "ZipDepthBaseNPU_weights",
            withExtension: "bin",
            subdirectory: "Models"
        ), let manifestURL = Bundle.module.url(
            forResource: "ZipDepthBaseNPU_weights",
            withExtension: "json",
            subdirectory: "Models"
        ) else
        {
            throw ZipDepthError("The bundled ZipDepth weights could not be found.")
        }

        try self.init(
            weightsBinaryURL: binaryURL,
            weightsManifestURL: manifestURL,
            inputWidth: inputWidth,
            inputHeight: inputHeight,
            commandQueue: commandQueue,
            maxFramesInFlight: maxFramesInFlight
        )
    }
}
