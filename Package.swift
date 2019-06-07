// swift-tools-version:5.0
import PackageDescription

let package = Package(
  name: "TransformerAPI",
  products: [
    .executable(name: "TransformerAPI", targets: ["TransformerAPI"]),
  ],
  dependencies: [
    .package(url: "https://github.com/apple/swift-nio.git", from: "2.0.0")
  ],
  targets: [
    .target(
      name: "TransformerAPI",
      dependencies: ["NIO", "NIOHTTP1"]),
  ]
)
