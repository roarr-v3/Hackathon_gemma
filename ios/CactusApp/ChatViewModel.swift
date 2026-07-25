import SwiftUI

struct ChatMessage: Identifiable, Equatable {
    enum Role {
        case user
        case assistant
    }

    enum Route: String {
        case phone = "On this iPhone"
        case computeNode = "Compute node"
    }

    let id = UUID()
    let role: Role
    let content: String
    let route: Route?
    let citations: [ComputeNodeCitation]

    init(
        role: Role,
        content: String,
        route: Route? = nil,
        citations: [ComputeNodeCitation] = []
    ) {
        self.role = role
        self.content = content
        self.route = route
        self.citations = citations
    }
}

struct AttachedDocument: Identifiable, Equatable {
    enum State: Equatable {
        case uploading
        case ready(remoteID: String)
        case failed(String)
    }

    let id: UUID
    let name: String
    var state: State

    init(name: String) {
        id = UUID()
        self.name = name
        state = .uploading
    }
}

@MainActor
final class ChatViewModel: ObservableObject {
    enum Status: Equatable {
        case loading
        case ready
        case failed(String)
    }

    @Published private(set) var messages: [ChatMessage] = []
    @Published private(set) var documents: [AttachedDocument] = []
    @Published private(set) var status: Status = .loading
    @Published private(set) var isGenerating = false

    private let inferenceService: any InferenceService
    private let computeNodeClient: ComputeNodeClient
    private var conversationID = UUID().uuidString
    private var isPreparing = false

    init(
        inferenceService: any InferenceService = CactusInferenceService(),
        computeNodeClient: ComputeNodeClient = ComputeNodeClient()
    ) {
        self.inferenceService = inferenceService
        self.computeNodeClient = computeNodeClient
    }

    var requiresComputeNode: Bool {
        !documents.isEmpty
    }

    var hasPendingDocument: Bool {
        documents.contains {
            if case .uploading = $0.state {
                return true
            }
            return false
        }
    }

    var hasFailedDocument: Bool {
        documents.contains {
            if case .failed = $0.state {
                return true
            }
            return false
        }
    }

    func prepare() async {
        guard status != .ready, !isPreparing else { return }
        isPreparing = true
        defer { isPreparing = false }
        status = .loading

        do {
            try await inferenceService.prepare()
            status = .ready
        } catch {
            status = .failed(error.localizedDescription)
        }
    }

    func attachDocument(at url: URL) async {
        var document = AttachedDocument(name: url.lastPathComponent)
        documents.append(document)

        do {
            let uploaded = try await computeNodeClient.uploadDocument(at: url)
            document.state = .ready(remoteID: uploaded.id)
        } catch {
            document.state = .failed(error.localizedDescription)
        }

        guard let index = documents.firstIndex(where: { $0.id == document.id }) else {
            return
        }
        documents[index] = document
    }

    func removeDocument(id: UUID) {
        documents.removeAll { $0.id == id }
    }

    func send(_ prompt: String) async -> String? {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            !trimmedPrompt.isEmpty,
            status == .ready,
            !isGenerating,
            !hasPendingDocument,
            !hasFailedDocument
        else {
            return nil
        }

        messages.append(ChatMessage(role: .user, content: trimmedPrompt))
        isGenerating = true
        defer { isGenerating = false }

        do {
            if requiresComputeNode {
                let documentIDs = documents.compactMap { document -> String? in
                    if case let .ready(remoteID) = document.state {
                        return remoteID
                    }
                    return nil
                }
                let response = try await computeNodeClient.generate(
                    prompt: trimmedPrompt,
                    documentIDs: documentIDs,
                    conversationID: conversationID
                )
                messages.append(
                    ChatMessage(
                        role: .assistant,
                        content: response.answer,
                        route: .computeNode,
                        citations: response.citations
                    )
                )
                return response.answer
            } else {
                let response = try await inferenceService.generate(prompt: trimmedPrompt)
                messages.append(
                    ChatMessage(
                        role: .assistant,
                        content: response,
                        route: .phone
                    )
                )
                return response
            }
        } catch {
            messages.append(
                ChatMessage(
                    role: .assistant,
                    content: "Inference failed: \(error.localizedDescription)",
                    route: requiresComputeNode ? .computeNode : .phone
                )
            )
            return nil
        }
    }

    func reset() async {
        await inferenceService.reset()
        messages.removeAll()
        documents.removeAll()
        conversationID = UUID().uuidString
    }

    func unload() async {
        await inferenceService.unload()
        messages.removeAll()
        documents.removeAll()
        status = .failed(InferenceError.modelNotDownloaded.localizedDescription)
    }
}
