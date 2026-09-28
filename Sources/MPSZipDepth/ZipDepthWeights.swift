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
        let byteRange = try self.byteRange(named: name)
        var values = [Float](repeating: 0, count: byteRange.count / MemoryLayout<Float>.stride)
        values.withUnsafeMutableBytes { destination in
            destination.copyBytes(from: self.data[byteRange])
        }
        return values
    }

    /// Copies the tensor's bytes straight out of the memory-mapped file into
    /// the graph constant -- no intermediate `[Float]`.
    func constant(_ graph: MPSGraph, named name: String) throws -> MPSGraphTensor
    {
        graph.constant(
            self.data.subdata(in: try self.byteRange(named: name)),
            shape: try self.shape(named: name).map(NSNumber.init(value:)),
            dataType: .float32
        )
    }

    /// The named float32 tensor's byte range within the mapped file.
    private func byteRange(named name: String) throws -> Range<Data.Index>
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

        let start = self.data.startIndex + byteOffset
        return start..<(start + byteCount)
    }
}
