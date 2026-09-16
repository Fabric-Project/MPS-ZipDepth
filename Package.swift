// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MPS-ZipDepth",
    platforms: [.macOS("15.0"), .iOS("18.0"), .visionOS("2.0")],
    products: [
        .library(name: "MPSZipDepth", targets: ["MPSZipDepth"]),
    ],
    targets: [
        .target(
            name: "MPSZipDepth",
            resources: [
                .copy("Models"),
                .copy("Utils/Compute"),
            ]
        ),
        .testTarget(
            name: "MPSZipDepthTests",
            dependencies: ["MPSZipDepth"]
        ),
    ],
    swiftLanguageVersions: [.v5]
)
