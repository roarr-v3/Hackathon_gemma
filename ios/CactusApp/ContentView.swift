import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var viewModel = ChatViewModel()
    @StateObject private var modelManager = ModelManager()
    @StateObject private var speechInput = SpeechInputService()
    @StateObject private var speechOutput = SpeechOutputService()
    @State private var prompt = ""
    @State private var selectedTab = 0
    @State private var isShowingDocumentPicker = false
    @State private var speaksReplies = true

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                VStack(spacing: 0) {
                    routeBanner
                    statusContent
                    attachedDocuments
                    composer
                }
                .navigationTitle("Gemma")
                .toolbar {
                    ToolbarItemGroup(placement: .topBarTrailing) {
                        Button {
                            speaksReplies.toggle()
                            if !speaksReplies {
                                speechOutput.stop()
                            }
                        } label: {
                            Image(
                                systemName: speaksReplies
                                    ? "speaker.wave.2.fill"
                                    : "speaker.slash.fill"
                            )
                        }
                        .accessibilityLabel(
                            speaksReplies ? "Disable spoken replies" : "Enable spoken replies"
                        )

                        if !viewModel.messages.isEmpty {
                            Button("New Chat", systemImage: "square.and.pencil") {
                                speechOutput.stop()
                                Task {
                                    await speechInput.stopListening()
                                    await viewModel.reset()
                                }
                            }
                        }
                    }
                }
                .fileImporter(
                    isPresented: $isShowingDocumentPicker,
                    allowedContentTypes: [.pdf, .plainText, .text, .json],
                    allowsMultipleSelection: false
                ) { result in
                    guard case let .success(urls) = result, let url = urls.first else {
                        return
                    }
                    Task { await viewModel.attachDocument(at: url) }
                }
            }
            .tabItem {
                Label("Chat", systemImage: "bubble.left.and.bubble.right")
            }
            .tag(0)

            ModelManagementView(
                modelManager: modelManager,
                unloadModel: { await viewModel.unload() }
            )
            .tabItem {
                Label("Models", systemImage: "internaldrive")
            }
            .tag(1)

            ComputeNodeSettingsView()
                .tabItem {
                    Label("Compute", systemImage: "desktopcomputer")
                }
                .tag(2)

#if DEBUG
            DiagnosticsView(modelManager: modelManager)
                .tabItem {
                    Label("Debug", systemImage: "gauge.with.dots.needle.67percent")
                }
                .tag(3)
#endif
        }
        .task {
            await viewModel.prepare()
        }
        .onChange(of: modelManager.state) { oldState, newState in
            if case .installed = newState, oldState != newState {
                Task { await viewModel.prepare() }
            }
        }
        .onChange(of: speechInput.transcript) { _, transcript in
            prompt = transcript
        }
        .onDisappear {
            speechOutput.stop()
            Task { await speechInput.stopListening() }
        }
    }

    private var routeBanner: some View {
        HStack(spacing: 8) {
            Image(
                systemName: viewModel.requiresComputeNode
                    ? "desktopcomputer"
                    : "iphone"
            )
            Text(
                viewModel.requiresComputeNode
                    ? "Document attached · compute node required"
                    : "Private inference on this iPhone"
            )
            .font(.subheadline.weight(.medium))
            Spacer()
        }
        .foregroundStyle(viewModel.requiresComputeNode ? Color.blue : Color.green)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.bar)
    }

    @ViewBuilder
    private var statusContent: some View {
        switch viewModel.status {
        case .loading:
            ContentUnavailableView {
                Label("Loading Gemma", systemImage: "cpu")
            } description: {
                Text("Preparing the on-device model. The first launch can take a moment.")
            } actions: {
                ProgressView()
            }
            .frame(maxHeight: .infinity)

        case let .failed(message):
            ContentUnavailableView {
                Label("Gemma Couldn’t Load", systemImage: "exclamationmark.triangle")
            } description: {
                Text(message)
            } actions: {
                if ModelStorage.isInstalled {
                    Button("Try Again") {
                        Task { await viewModel.prepare() }
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    Button("Manage Models") {
                        selectedTab = 1
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .frame(maxHeight: .infinity)

        case .ready:
            chatContent
        }
    }

    @ViewBuilder
    private var attachedDocuments: some View {
        if !viewModel.documents.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(viewModel.documents) { document in
                        HStack(spacing: 7) {
                            documentStatusIcon(document.state)

                            Text(document.name)
                                .font(.caption)
                                .lineLimit(1)

                            Button {
                                viewModel.removeDocument(id: document.id)
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("Remove \(document.name)")
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(Color.blue.opacity(0.12), in: Capsule())
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
            }
            .background(.bar)
        }
    }

    @ViewBuilder
    private var chatContent: some View {
        if viewModel.messages.isEmpty {
            ContentUnavailableView {
                Label("Chat with Gemma", systemImage: "bubble.left.and.bubble.right")
            } description: {
                Text("Private, on-device inference powered by Cactus.")
            }
            .frame(maxHeight: .infinity)
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 14) {
                        ForEach(viewModel.messages) { message in
                            messageBubble(message)
                                .id(message.id)
                        }

                        if viewModel.isGenerating {
                            HStack {
                                ProgressView()
                                Text("Gemma is thinking…")
                                    .foregroundStyle(.secondary)
                                Spacer()
                            }
                            .padding(.horizontal)
                        }
                    }
                    .padding(.vertical)
                }
                .onChange(of: viewModel.messages) { _, messages in
                    guard let lastMessage = messages.last else { return }
                    withAnimation {
                        proxy.scrollTo(lastMessage.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            if speechInput.isListening {
                Label("Listening on this iPhone… Tap the microphone to stop.", systemImage: "waveform")
                    .font(.caption)
                    .foregroundStyle(.red)
            } else if speechInput.isStarting {
                Label("Preparing on-device speech recognition…", systemImage: "waveform")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if let errorMessage = speechInput.errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            HStack(alignment: .bottom, spacing: 12) {
                Button {
                    isShowingDocumentPicker = true
                } label: {
                    Image(systemName: "paperclip")
                }
                .buttonStyle(.bordered)
                .disabled(
                    viewModel.isGenerating
                        || viewModel.hasPendingDocument
                        || speechInput.isListening
                        || speechInput.isStarting
                )
                .accessibilityLabel("Attach a document")

                TextField("Message Gemma…", text: $prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...5)
                    .disabled(speechInput.isListening || speechInput.isStarting)
                    .onSubmit(send)

                Button(action: toggleListening) {
                    Group {
                        if speechInput.isStarting {
                            ProgressView()
                        } else {
                            Image(
                                systemName: speechInput.isListening
                                    ? "stop.fill"
                                    : "mic.fill"
                            )
                        }
                    }
                    .frame(width: 18, height: 18)
                }
                .buttonStyle(.bordered)
                .tint(speechInput.isListening ? .red : .accentColor)
                .disabled(
                    viewModel.status != .ready
                        || viewModel.isGenerating
                        || viewModel.hasPendingDocument
                        || viewModel.hasFailedDocument
                        || speechInput.isStarting
                )
                .accessibilityLabel(
                    speechInput.isListening ? "Stop listening" : "Start listening"
                )

                Button(action: send) {
                    Image(systemName: "arrow.up")
                        .fontWeight(.semibold)
                }
                .buttonStyle(.borderedProminent)
                .disabled(
                    prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        || viewModel.status != .ready
                        || viewModel.isGenerating
                        || viewModel.hasPendingDocument
                        || viewModel.hasFailedDocument
                        || speechInput.isListening
                        || speechInput.isStarting
                )
                .accessibilityLabel("Send message")
            }
        }
        .padding()
        .background(.bar)
    }

    private func messageBubble(_ message: ChatMessage) -> some View {
        HStack {
            if message.role == .user {
                Spacer(minLength: 48)
            }

            VStack(alignment: .leading, spacing: 7) {
                Text(message.content)
                    .textSelection(.enabled)

                if let route = message.route {
                    Label(
                        route.rawValue,
                        systemImage: route == .phone ? "iphone" : "desktopcomputer"
                    )
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                }

                if !message.citations.isEmpty {
                    Divider()
                    ForEach(message.citations, id: \.chunkID) { citation in
                        Label(citation.documentName, systemImage: "doc.text")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .foregroundStyle(message.role == .user ? Color.white : Color.primary)
            .background(
                message.role == .user ? Color.accentColor : Color.secondary.opacity(0.15),
                in: RoundedRectangle(cornerRadius: 18)
            )

            if message.role == .assistant {
                Spacer(minLength: 48)
            }
        }
        .padding(.horizontal)
    }

    @ViewBuilder
    private func documentStatusIcon(_ state: AttachedDocument.State) -> some View {
        switch state {
        case .uploading:
            ProgressView()
                .controlSize(.small)
        case .ready:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
        }
    }

    private func send() {
        let submittedPrompt = prompt
        prompt = ""
        speechOutput.stop()
        Task {
            if let response = await viewModel.send(submittedPrompt), speaksReplies {
                speechOutput.speak(response)
            }
        }
    }

    private func toggleListening() {
        if speechInput.isListening {
            Task {
                await speechInput.stopListening()
            }
        } else {
            speechOutput.stop()
            speechInput.clearError()
            prompt = ""
            Task {
                await speechInput.startListening()
            }
        }
    }
}

#Preview {
    ContentView()
}
