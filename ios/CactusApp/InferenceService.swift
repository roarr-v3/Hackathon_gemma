import Foundation
import Darwin
import cactus

protocol InferenceService: Sendable {
    func prepare() async throws
    func generate(prompt: String) async throws -> String
    func reset() async
    func unload() async
}

enum InferenceError: LocalizedError {
    case modelNotDownloaded
    case modelInitializationFailed
    case requestEncodingFailed
    case completionFailed(code: Int32)
    case invalidResponse
    case runtime(String)

    var errorDescription: String? {
        switch self {
        case .modelNotDownloaded:
            "Download Gemma from the Models tab first."
        case .modelInitializationFailed:
            "Cactus could not load the Gemma model."
        case .requestEncodingFailed:
            "The chat request could not be encoded."
        case let .completionFailed(code):
            "Cactus inference failed with code \(code)."
        case .invalidResponse:
            "Cactus returned an invalid response."
        case let .runtime(message):
            message
        }
    }
}

actor CactusInferenceService: InferenceService {
    private struct RequestMessage: Codable {
        let role: String
        let content: String
    }

    private struct GenerationOptions: Encodable {
        let maxTokens = 512
        let temperature = 0.7
        let topP = 0.95
        let repetitionPenalty = 1.1
        let autoHandoff = false
        let enableThinkingIfSupported = false

        enum CodingKeys: String, CodingKey {
            case maxTokens = "max_tokens"
            case temperature
            case topP = "top_p"
            case repetitionPenalty = "repetition_penalty"
            case autoHandoff = "auto_handoff"
            case enableThinkingIfSupported = "enable_thinking_if_supported"
        }
    }

    private static let responseBufferSize = 1_048_576

    private var model: cactus_model_t?
    private var conversation = [
        RequestMessage(
            role: "system",
            content: "You are Gemma, a helpful assistant running privately on the user's device."
        )
    ]

    deinit {
        if let model {
            cactus_destroy(model)
        }
    }

    func prepare() async throws {
        guard model == nil else { return }
        setenv("CACTUS_NO_CLOUD_TELE", "1", 1)

        guard ModelStorage.isInstalled else {
            throw InferenceError.modelNotDownloaded
        }

        let loadedModel = ModelStorage.modelURL.path.withCString {
            cactus_init($0, nil, false)
        }

        guard let loadedModel else {
            throw InferenceError.modelInitializationFailed
        }
        model = loadedModel
    }

    func generate(prompt: String) async throws -> String {
        try await prepare()
        guard let model else {
            throw InferenceError.modelInitializationFailed
        }

        conversation.append(RequestMessage(role: "user", content: prompt))

        do {
            let messagesData = try JSONEncoder().encode(conversation)
            let optionsData = try JSONEncoder().encode(GenerationOptions())
            guard
                let messagesJSON = String(data: messagesData, encoding: .utf8),
                let optionsJSON = String(data: optionsData, encoding: .utf8)
            else {
                throw InferenceError.requestEncodingFailed
            }

            var responseBuffer = [CChar](
                repeating: 0,
                count: Self.responseBufferSize
            )

            let result = messagesJSON.withCString { messagesPointer in
                optionsJSON.withCString { optionsPointer in
                    responseBuffer.withUnsafeMutableBufferPointer { buffer in
                        cactus_complete(
                            model,
                            messagesPointer,
                            buffer.baseAddress,
                            buffer.count,
                            optionsPointer,
                            nil,
                            nil,
                            nil,
                            nil,
                            0
                        )
                    }
                }
            }

            guard result >= 0 else {
                throw InferenceError.completionFailed(code: result)
            }

            let responseJSON = String(cString: responseBuffer)
            let answer = try Self.extractAnswer(from: responseJSON)
            conversation.append(RequestMessage(role: "assistant", content: answer))
            return answer
        } catch {
            if conversation.last?.role == "user" {
                conversation.removeLast()
            }
            throw error
        }
    }

    func reset() async {
        if let model {
            cactus_reset(model)
        }
        conversation = [
            RequestMessage(
                role: "system",
                content: "You are Gemma, a helpful assistant running privately on the user's device."
            )
        ]
    }

    func unload() async {
        if let model {
            cactus_destroy(model)
            self.model = nil
        }
        conversation = [
            RequestMessage(
                role: "system",
                content: "You are Gemma, a helpful assistant running privately on the user's device."
            )
        ]
    }

    private static func extractAnswer(from responseJSON: String) throws -> String {
        guard
            let data = responseJSON.data(using: .utf8),
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw InferenceError.invalidResponse
        }

        if object["success"] as? Bool == false {
            let message = object["error"] as? String ?? "Cactus reported an inference error."
            throw InferenceError.runtime(message)
        }

        guard let response = object["response"] as? String else {
            throw InferenceError.invalidResponse
        }
        return response
    }

}
