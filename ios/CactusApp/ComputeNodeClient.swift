import Foundation

enum ComputeNodeError: LocalizedError {
    case notConfigured
    case invalidResponse
    case server(status: Int, message: String)

    var errorDescription: String? {
        switch self {
        case .notConfigured:
            "Configure the Mac compute node before attaching a document."
        case .invalidResponse:
            "The compute node returned an invalid response."
        case let .server(status, message):
            "Compute node error \(status): \(message)"
        }
    }
}

struct ComputeNodeHealth: Decodable {
    let status: String
    let serverName: String
    let inferenceReady: Bool
    let model: String

    enum CodingKeys: String, CodingKey {
        case status
        case serverName = "server_name"
        case inferenceReady = "inference_ready"
        case model
    }
}

struct RemoteDocument: Decodable {
    let id: String
    let name: String
    let characterCount: Int

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case characterCount = "character_count"
    }
}

struct ComputeNodeCitation: Decodable, Equatable {
    let documentID: String
    let documentName: String
    let chunkID: String

    enum CodingKeys: String, CodingKey {
        case documentID = "document_id"
        case documentName = "document_name"
        case chunkID = "chunk_id"
    }
}

struct ComputeNodeAnswer: Decodable {
    let answer: String
    let model: String
    let citations: [ComputeNodeCitation]
}

actor ComputeNodeClient {
    private struct ChatRequest: Encodable {
        let conversationID: String
        let message: String
        let documentIDs: [String]
        let adapterID: String?

        enum CodingKeys: String, CodingKey {
            case conversationID = "conversation_id"
            case message
            case documentIDs = "document_ids"
            case adapterID = "adapter_id"
        }
    }

    private let session: URLSession

    init(session: URLSession = .shared) {
        self.session = session
    }

    func health() async throws -> ComputeNodeHealth {
        let request = try makeRequest(path: "v1/health", method: "GET")
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(ComputeNodeHealth.self, from: data)
    }

    func uploadDocument(at fileURL: URL) async throws -> RemoteDocument {
        let hasScopedAccess = fileURL.startAccessingSecurityScopedResource()
        defer {
            if hasScopedAccess {
                fileURL.stopAccessingSecurityScopedResource()
            }
        }

        let fileData = try Data(contentsOf: fileURL)
        let boundary = "GemmaMesh-\(UUID().uuidString)"
        var body = Data()
        body.appendUTF8("--\(boundary)\r\n")
        body.appendUTF8(
            "Content-Disposition: form-data; name=\"file\"; filename=\"\(fileURL.lastPathComponent.escapedMultipartFilename)\"\r\n"
        )
        body.appendUTF8("Content-Type: application/octet-stream\r\n\r\n")
        body.append(fileData)
        body.appendUTF8("\r\n--\(boundary)--\r\n")

        var request = try makeRequest(path: "v1/documents", method: "POST")
        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        request.httpBody = body

        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(RemoteDocument.self, from: data)
    }

    func generate(
        prompt: String,
        documentIDs: [String],
        adapterID: String? = nil,
        conversationID: String
    ) async throws -> ComputeNodeAnswer {
        var request = try makeRequest(path: "v1/chat/completions", method: "POST")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(
            ChatRequest(
                conversationID: conversationID,
                message: prompt,
                documentIDs: documentIDs,
                adapterID: adapterID
            )
        )

        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try JSONDecoder().decode(ComputeNodeAnswer.self, from: data)
    }

    private func makeRequest(path: String, method: String) throws -> URLRequest {
        guard let baseURL = ComputeNodeConfiguration.baseURL else {
            throw ComputeNodeError.notConfigured
        }
        let url = baseURL.appendingPathComponent(path)
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = 120
        let token = ComputeNodeConfiguration.apiToken
        if !token.isEmpty {
            request.setValue(token, forHTTPHeaderField: "X-API-Key")
        }
        return request
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw ComputeNodeError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            let detail = object?["detail"] as? String
            let fallback = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw ComputeNodeError.server(
                status: httpResponse.statusCode,
                message: detail ?? fallback
            )
        }
    }
}

private extension Data {
    mutating func appendUTF8(_ string: String) {
        append(Data(string.utf8))
    }
}

private extension String {
    var escapedMultipartFilename: String {
        replacingOccurrences(of: "\"", with: "")
            .replacingOccurrences(of: "\r", with: "")
            .replacingOccurrences(of: "\n", with: "")
    }
}
