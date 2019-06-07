// Modified from https://github.com/apple/swift-nio/blob/master/Sources/NIOHTTP1Server/main.swift
import NIO
import NIOHTTP1

// https://github.com/tensorflow/swift-models/tree/master/Transformer
import Python
import TensorFlow
import Foundation


func transform(_ content: String) -> String {
    // return "You sent \(content)"
    do {
        let modelName = "117M"
        let sys = Python.import("sys")
        sys.path = sys.path + [".", "./Sources/TransformerAPI"]
        let encoder = Python.import("encoder").get_encoder(modelName)

        let checkpoint = "./Sources/TransformerAPI/models/\(modelName)/model.ckpt"
        let configFile = "./Sources/TransformerAPI/models/\(modelName)/hparams.json"
        let configData = try Data(contentsOf: URL(fileURLWithPath: configFile))
        let config = try JSONDecoder().decode(Config.self, from: configData)
        let model = TransformerLM(
          contentsOfPythonCheckpointFile: checkpoint, config: config, scope: "model")

        let start_token = Int32(encoder.encoder["<|endoftext|>"])!
        var tokens = Tensor(shape: [1, 1], scalars: [start_token])
        let temperature = Float(0.5)

        let pytok = encoder.encode(content)
        let tokarr: [Int32] = Array<Int>(pytok)!.map { Int32($0) }
        tokens = Tensor(shape: [1, tokarr.count], scalars: tokarr)

        let empty = Tensor<Float>(zeros: [config.headCount, 0, config.embeddingSize / config.headCount])
        var states = (0..<config.layerCount).map { _ in AttentionContext(key: empty, value: empty) }

        var result = ""

        for _ in 0..<100 {
            let logits = model(tokens, states: &states)
            let (batchSize, timeSteps, vocabSize) = (logits.shape[0], logits.shape[1], logits.shape[2])
            let lastLogit = logits.slice(
              lowerBounds: [0, timeSteps - 1, 0],
              upperBounds: [batchSize, timeSteps, vocabSize]) / temperature
            tokens = Raw.multinomial(logits: lastLogit.squeezingShape(at: 1), numSamples: Tensor<Int32>(1))
            let chunkOfText = encoder.decode(tokens[0].makeNumpyArray())
            result += String(chunkOfText) ?? ""
        }

        return result
    }
    catch {
        return "INVALID CONFIGURATION"
    }
}


private func httpResponseHead(request: HTTPRequestHead, status: HTTPResponseStatus, headers: HTTPHeaders = HTTPHeaders()) -> HTTPResponseHead {
    var head = HTTPResponseHead(version: request.version, status: status, headers: headers)
    let connectionHeaders: [String] = head.headers[canonicalForm: "connection"].map { $0.lowercased() }

    if !connectionHeaders.contains("keep-alive") && !connectionHeaders.contains("close") {
        // the user hasn't pre-set either 'keep-alive' or 'close', so we might need to add headers

        switch (request.isKeepAlive, request.version.major, request.version.minor) {
        case (true, 1, 0):
            // HTTP/1.0 and the request has 'Connection: keep-alive', we should mirror that
            head.headers.add(name: "Connection", value: "keep-alive")
        case (false, 1, let n) where n >= 1:
            // HTTP/1.1 (or treated as such) and the request has 'Connection: close', we should mirror that
            head.headers.add(name: "Connection", value: "close")
        default:
            // we should match the default or are dealing with some HTTP that we don't support, let's leave as is
            ()
        }
    }
    return head
}

private final class HTTPHandler: ChannelInboundHandler {
    public typealias InboundIn = HTTPServerRequestPart
    public typealias OutboundOut = HTTPServerResponsePart

    private var buffer: ByteBuffer! = nil
    private var keepAlive = false

    private var infoSavedRequestHead: HTTPRequestHead?
    private var infoSavedBodyBytes: Int = 0
    private var infoSavedBodyByteBuffer: ByteBuffer? = nil

    private var continuousCount: Int = 0

    private var handler: ((ChannelHandlerContext, HTTPServerRequestPart) -> Void)?
    private var handlerFuture: EventLoopFuture<Void>?
    private let defaultResponse = "Hello World\r\n"

    public init() {
    }

    func blabber(context: ChannelHandlerContext, request: HTTPServerRequestPart) {
        switch request {
        case .head(let request):
            self.infoSavedRequestHead = request
            self.infoSavedBodyBytes = 0
            self.keepAlive = request.isKeepAlive
        case .body(buffer: var buf):
            self.infoSavedBodyBytes += buf.readableBytes
            if self.infoSavedBodyByteBuffer == nil {
                self.infoSavedBodyByteBuffer = buf
            } else {
                self.infoSavedBodyByteBuffer!.writeBuffer(&buf)
            }
        case .end:

            let infoSavedBodyByteBufferLength = self.infoSavedBodyByteBuffer?.readableBytes ?? 0
            if let inputText = self.infoSavedBodyByteBuffer?.readString(length: infoSavedBodyByteBufferLength) {
                print("Input : \(inputText)")
                let response = transform(inputText)
                print("Output: \(response)")
                
                self.buffer.clear()
                self.buffer.writeString(response)
                var headers = HTTPHeaders()
                headers.add(name: "Content-Length", value: "\(response.utf8.count)")
                context.write(self.wrapOutboundOut(.head(httpResponseHead(request: self.infoSavedRequestHead!, status: .ok, headers: headers))), promise: nil)
                context.write(self.wrapOutboundOut(.body(.byteBuffer(self.buffer))), promise: nil)
                self.completeResponse(context, trailers: nil, promise: nil)
            } else {
                // TODO Change status code
                let response = "INVALID INPUT"
                
                self.buffer.clear()
                self.buffer.writeString(response)
                var headers = HTTPHeaders()
                headers.add(name: "Content-Length", value: "\(response.utf8.count)")
                context.write(self.wrapOutboundOut(.head(httpResponseHead(request: self.infoSavedRequestHead!, status: .ok, headers: headers))), promise: nil)
                context.write(self.wrapOutboundOut(.body(.byteBuffer(self.buffer))), promise: nil)
                self.completeResponse(context, trailers: nil, promise: nil)
            }
        }
    }

    private func completeResponse(_ context: ChannelHandlerContext, trailers: HTTPHeaders?, promise: EventLoopPromise<Void>?) {
        let promise = self.keepAlive ? promise : (promise ?? context.eventLoop.makePromise())
        if !self.keepAlive {
            promise!.futureResult.whenComplete { (_: Result<Void, Error>) in context.close(promise: nil) }
        }
        self.handler = nil

        context.writeAndFlush(self.wrapOutboundOut(.end(trailers)), promise: promise)
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let reqPart = self.unwrapInboundIn(data)

        self.blabber(context: context, request: reqPart)
    }

    func channelReadComplete(context: ChannelHandlerContext) {
        context.flush()
    }

    func handlerAdded(context: ChannelHandlerContext) {
        self.buffer = context.channel.allocator.buffer(capacity: 0)
    }
}

// First argument is the program path
var arguments = CommandLine.arguments.dropFirst(0) // just to get an ArraySlice<String> from [String]
var allowHalfClosure = true
if arguments.dropFirst().first == .some("--disable-half-closure") {
    allowHalfClosure = false
    arguments = arguments.dropFirst()
}
let arg1 = arguments.dropFirst().first
let arg2 = arguments.dropFirst(2).first
let arg3 = arguments.dropFirst(3).first

let defaultHost = "::1"
let defaultPort = 8888
let defaultHtdocs = "/dev/null/"

enum BindTo {
    case ip(host: String, port: Int)
    case unixDomainSocket(path: String)
}

let htdocs: String
let bindTarget: BindTo

switch (arg1, arg1.flatMap(Int.init), arg2, arg2.flatMap(Int.init), arg3) {
case (.some(let h), _ , _, .some(let p), let maybeHtdocs):
    /* second arg an integer --> host port [htdocs] */
    bindTarget = .ip(host: h, port: p)
    htdocs = maybeHtdocs ?? defaultHtdocs
case (_, .some(let p), let maybeHtdocs, _, _):
    /* first arg an integer --> port [htdocs] */
    bindTarget = .ip(host: defaultHost, port: p)
    htdocs = maybeHtdocs ?? defaultHtdocs
case (.some(let portString), .none, let maybeHtdocs, .none, .none):
    /* couldn't parse as number --> uds-path [htdocs] */
    bindTarget = .unixDomainSocket(path: portString)
    htdocs = maybeHtdocs ?? defaultHtdocs
default:
    htdocs = defaultHtdocs
    bindTarget = BindTo.ip(host: defaultHost, port: defaultPort)
}

let group = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)

let bootstrap = ServerBootstrap(group: group)
    // Specify backlog and enable SO_REUSEADDR for the server itself
    .serverChannelOption(ChannelOptions.backlog, value: 256)
    .serverChannelOption(ChannelOptions.socket(SocketOptionLevel(SOL_SOCKET), SO_REUSEADDR), value: 1)

    // Set the handlers that are applied to the accepted Channels
    .childChannelInitializer { channel in
        channel.pipeline.configureHTTPServerPipeline(withErrorHandling: true).flatMap {
            channel.pipeline.addHandler(HTTPHandler())
        }
    }

    // Enable TCP_NODELAY and SO_REUSEADDR for the accepted Channels
    .childChannelOption(ChannelOptions.socket(IPPROTO_TCP, TCP_NODELAY), value: 1)
    .childChannelOption(ChannelOptions.socket(SocketOptionLevel(SOL_SOCKET), SO_REUSEADDR), value: 1)
    .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 1)
    .childChannelOption(ChannelOptions.allowRemoteHalfClosure, value: allowHalfClosure)

defer {
    try! group.syncShutdownGracefully()
}

print("htdocs = \(htdocs)")

let channel = try { () -> Channel in
    switch bindTarget {
    case .ip(let host, let port):
        return try bootstrap.bind(host: host, port: port).wait()
    case .unixDomainSocket(let path):
        return try bootstrap.bind(unixDomainSocketPath: path).wait()
    }
}()

guard let localAddress = channel.localAddress else {
    fatalError("Address was unable to bind. Please check that the socket was not closed or that the address family was understood.")
}
print("Server started and listening on \(localAddress), htdocs path \(htdocs)")

// This will never unblock as we don't close the ServerChannel
try channel.closeFuture.wait()

print("Server closed")
