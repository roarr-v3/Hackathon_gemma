import SwiftUI

struct ChatMessage: Identifiable, Equatable {
    enum Role {
        case user
        case assistant
    }

    enum Route: String, CaseIterable, Identifiable {
        case phone
        case computeNode

        var id: String { rawValue }

        var modelName: String {
            switch self {
            case .phone:
                "Gemma 4 E2B"
            case .computeNode:
                "Gemma 4 E4B"
            }
        }

        var locationName: String {
            switch self {
            case .phone:
                "This iPhone"
            case .computeNode:
                "Mac compute"
            }
        }

        var messageLabel: String {
            switch self {
            case .phone:
                "On this iPhone"
            case .computeNode:
                "Compute node"
            }
        }

        var systemImage: String {
            switch self {
            case .phone:
                "iphone"
            case .computeNode:
                "desktopcomputer"
            }
        }
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

struct ConversationTrainingArtifact: Identifiable {
    let id = UUID()
    let url: URL
    let contents: String
}

enum ConversationTrainingError: LocalizedError {
    case conversationIsEmpty

    var errorDescription: String? {
        switch self {
        case .conversationIsEmpty:
            "Have a conversation before creating a training summary."
        }
    }
}

@MainActor
final class ChatViewModel: ObservableObject {
    private static let selectedRouteKey = "chat.selectedInferenceRoute"

    enum Status: Equatable {
        case loading
        case ready
        case failed(String)
    }

    @Published private(set) var messages: [ChatMessage] = []
    @Published private(set) var documents: [AttachedDocument] = []
    @Published private(set) var status: Status = .loading
    @Published private(set) var isGenerating = false
    @Published private(set) var selectedRoute: ChatMessage.Route

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
        selectedRoute = ChatMessage.Route(
            rawValue: UserDefaults.standard.string(forKey: Self.selectedRouteKey) ?? ""
        ) ?? .phone
    }

    var requiresComputeNode: Bool {
        !documents.isEmpty
    }

    var activeRoute: ChatMessage.Route {
        requiresComputeNode ? .computeNode : selectedRoute
    }

    var canUseActiveRoute: Bool {
        activeRoute == .computeNode || status == .ready
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

    func selectRoute(_ route: ChatMessage.Route) {
        selectedRoute = route
        UserDefaults.standard.set(route.rawValue, forKey: Self.selectedRouteKey)
    }

    func attachDocument(at url: URL) async {
        selectRoute(.computeNode)
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
            canUseActiveRoute,
            !isGenerating,
            !hasPendingDocument,
            !hasFailedDocument
        else {
            return nil
        }

        messages.append(ChatMessage(role: .user, content: trimmedPrompt))
        isGenerating = true
        defer { isGenerating = false }

        let route = activeRoute

        do {
            if route == .computeNode {
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
                    route: route
                )
            )
            return nil
        }
    }

    func createTrainingSummary() async throws -> ConversationTrainingArtifact {
        guard !messages.isEmpty else {
            throw ConversationTrainingError.conversationIsEmpty
        }
        guard !isGenerating else {
            throw InferenceError.runtime("Wait for the current reply to finish first.")
        }

        let route = activeRoute
        let transcript = messages.map { message in
            let speaker = message.role == .user ? "USER" : "ASSISTANT"
            return "\(speaker): \(message.content)"
        }
        .joined(separator: "\n\n")

        let boundedTranscript = String(transcript.suffix(32_000))
        let summaryPrompt = """
        Create a precise training brief from the conversation below.

        Rules:
        - Preserve only information supported by the conversation.
        - Do not invent facts or infer sensitive attributes.
        - Keep exact names, requirements, decisions, constraints, and corrections.
        - Capture the user's communication and response-style preferences.
        - Include reusable instruction-response examples when the conversation supports them.
        - Remove greetings, repetition, and debugging noise.
        - Write plain text using exactly these headings:
          PURPOSE
          VERIFIED FACTS
          USER PREFERENCES
          DECISIONS AND CONSTRAINTS
          REUSABLE INSTRUCTION-RESPONSE EXAMPLES
          OPEN QUESTIONS

        CONVERSATION:
        \(boundedTranscript)
        """

        isGenerating = true
        defer { isGenerating = false }

        let summary: String
        if route == .computeNode {
            let response = try await computeNodeClient.generate(
                prompt: summaryPrompt,
                documentIDs: [],
                conversationID: "training-summary-\(UUID().uuidString)"
            )
            summary = response.answer
        } else {
            summary = try await inferenceService.generate(prompt: summaryPrompt)
        }

        let contents = """
        GEMMA CONVERSATION TRAINING BRIEF
        Generated: \(Date().formatted(date: .numeric, time: .standard))
        Generated with: \(route.modelName) — \(route.locationName)

        \(summary.trimmingCharacters(in: .whitespacesAndNewlines))

        SOURCE TRANSCRIPT
        \(boundedTranscript)
        """

        let fileManager = FileManager.default
        let directory = try fileManager.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        .appendingPathComponent("Training Summaries", isDirectory: true)
        try fileManager.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd-HHmmss"
        let fileURL = directory.appendingPathComponent(
            "gemma-training-brief-\(formatter.string(from: Date())).txt"
        )
        try contents.write(to: fileURL, atomically: true, encoding: .utf8)

        return ConversationTrainingArtifact(url: fileURL, contents: contents)
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
