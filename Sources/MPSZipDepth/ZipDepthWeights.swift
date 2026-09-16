import Foundation
import MetalPerformanceShadersGraph

final class ZipDepthWeights
{
    private struct Entry: Decodable
    {
        let offset: Int
        let count: Int
        let shape: [Int]
        let dtype: String
    }

    private let entries: [String: Entry]
    private let data: Data

    init(binaryURL: URL, manifestURL: URL) throws
    {
        self.data = try Data(contentsOf: binaryURL, options: .mappedIfSafe)
        self.entries = try JSONDecoder().decode(
            [String: Entry].self,
            from: Data(contentsOf: manifestURL)
        )
    }

    func shape(named name: String) throws -> [Int]
    {
        guard let entry = self.entries[name] else
        {
            throw ZipDepthError("Missing ZipDepth tensor '\(name)'.")
        }
        return entry.shape
    }

    func floats(named name: String) throws -> [Float]
    {
        guard let entry = self.entries[name] else
        {
            throw ZipDepthError("Missing ZipDepth tensor '\(name)'.")
        }
        guard entry.dtype == "float32" else
        {
            throw ZipDepthError("ZipDepth tensor '\(name)' has unsupported dtype '\(entry.dtype)'.")
        }

        let byteOffset = entry.offset * MemoryLayout<Float>.stride
        let byteCount = entry.count * MemoryLayout<Float>.stride
        guard byteOffset >= 0, byteCount >= 0, byteOffset + byteCount <= self.data.count else
        {
            throw ZipDepthError("ZipDepth tensor '\(name)' exceeds the weights file.")
        }

        var values = [Float](repeating: 0, count: entry.count)
        self.data.withUnsafeBytes { sourceBuffer in
            guard let source = sourceBuffer.baseAddress?.advanced(by: byteOffset) else { return }
            values.withUnsafeMutableBytes { destination in
                destination.copyMemory(from: UnsafeRawBufferPointer(start: source, count: byteCount))
            }
        }
        return values
    }

    func constant(_ graph: MPSGraph, named name: String) throws -> MPSGraphTensor
    {
        let values = try self.floats(named: name)
        return graph.constant(
            Data(bytes: values, count: values.count * MemoryLayout<Float>.stride),
            shape: try self.shape(named: name).map(NSNumber.init(value:)),
            dataType: .float32
        )
    }
}
