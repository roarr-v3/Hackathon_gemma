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
    @State private var shouldSpeakNextReply = false
    @State private var isCreatingAdapter = false
    @State private var adapterArtifact: ConversationTrainingArtifact?
    @State private var adapterError: String?
    @FocusState private var isPromptFocused: Bool
    @Namespace private var glassNamespace

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack {
                ZStack {
                    appBackground

                    VStack(spacing: 0) {
                        routeBanner
                        statusContent
                        attachedDocuments
                        composer
                    }
                    .simultaneousGesture(
                        DragGesture(minimumDistance: 18)
                            .onEnded { value in
                                let verticalDistance = value.translation.height
                                let horizontalDistance = abs(value.translation.width)
                                if verticalDistance > 28,
                                   verticalDistance > horizontalDistance {
                                    isPromptFocused = false
                                }
                            }
                    )
                }
                .navigationTitle("")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .principal) {
                        modelSelector
                    }

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
            if viewModel.selectedRoute == .phone {
                await viewModel.prepare()
            }
        }
        .onChange(of: modelManager.state) { oldState, newState in
            if case .installed = newState, oldState != newState {
                Task { await viewModel.prepare() }
            }
        }
        .onChange(of: speechInput.transcript) { _, transcript in
            prompt = transcript
            if !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                shouldSpeakNextReply = true
            }
        }
        .onDisappear {
            speechOutput.stop()
            Task { await speechInput.stopListening() }
        }
        .sheet(item: $adapterArtifact) { artifact in
            adapterReadySheet(artifact)
        }
        .alert(
            "Couldn’t Create Adapter",
            isPresented: Binding(
                get: { adapterError != nil },
                set: { if !$0 { adapterError = nil } }
            )
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(adapterError ?? "Please try again.")
        }
    }

    private var routeBanner: some View {
        HStack(spacing: 7) {
            Image(systemName: viewModel.activeRoute.systemImage)
                .symbolRenderingMode(.hierarchical)
            Text(routeDescription)
                .font(.caption.weight(.semibold))
            Image(systemName: "lock.fill")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 13)
        .padding(.vertical, 9)
        .glassEffect(
            .regular.tint(
                viewModel.activeRoute == .computeNode
                    ? Color.indigo.opacity(0.45)
                    : Color.cyan.opacity(0.35)
            ),
            in: .capsule
        )
        .padding(.top, 8)
        .padding(.horizontal)
    }

    private var modelSelector: some View {
        Menu {
            Section("Select model") {
                ForEach(ChatMessage.Route.allCases) { route in
                    Button {
                        viewModel.selectRoute(route)
                        if route == .phone {
                            Task { await viewModel.prepare() }
                        }
                    } label: {
                        Label(
                            "\(route.modelName) · \(route.locationName)",
                            systemImage: viewModel.activeRoute == route
                                ? "checkmark.circle.fill"
                                : route.systemImage
                        )
                    }
                }
            }

            Section {
                Button {
                    selectedTab = 2
                } label: {
                    Label("Compute node settings", systemImage: "gear")
                }

                if viewModel.requiresComputeNode {
                    Label("Documents require Mac compute", systemImage: "doc.text")
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: viewModel.activeRoute.systemImage)
                    .font(.subheadline)
                Text(viewModel.activeRoute.modelName)
                    .font(.headline)
                Image(systemName: "chevron.down")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(.secondary)
            }
            .foregroundStyle(.primary)
            .contentShape(Rectangle())
        }
        .accessibilityLabel(
            "Selected model: \(viewModel.activeRoute.modelName), \(viewModel.activeRoute.locationName)"
        )
    }

    private var routeDescription: String {
        if viewModel.requiresComputeNode {
            return "Document attached · Mac compute required"
        }
        switch viewModel.activeRoute {
        case .phone:
            return "Private inference on this iPhone"
        case .computeNode:
            return "Private inference on your Mac"
        }
    }

    @ViewBuilder
    private var statusContent: some View {
        if viewModel.activeRoute == .computeNode {
            chatContent
        } else {
            localModelStatusContent
        }
    }

    @ViewBuilder
    private var localModelStatusContent: some View {
        switch viewModel.status {
        case .loading:
            VStack(spacing: 16) {
                Image(systemName: "cpu")
                    .font(.system(size: 28, weight: .semibold))
                    .frame(width: 64, height: 64)
                    .glassEffect(.regular.tint(Color.cyan.opacity(0.3)), in: .circle)

                Text("Loading Gemma")
                    .font(.title3.bold())
                Text("Preparing the on-device model. The first launch can take a moment.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                ProgressView()
            }
            .padding(28)
            .glassEffect(.clear, in: .rect(cornerRadius: 28))
            .padding()
            .frame(maxHeight: .infinity)

        case let .failed(message):
            VStack(spacing: 16) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 26, weight: .semibold))
                    .foregroundStyle(.orange)
                    .frame(width: 64, height: 64)
                    .glassEffect(.regular.tint(Color.orange.opacity(0.25)), in: .circle)

                Text("Gemma Couldn’t Load")
                    .font(.title3.bold())
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                if ModelStorage.isInstalled {
                    Button("Try Again") {
                        Task { await viewModel.prepare() }
                    }
                    .buttonStyle(.glassProminent)
                } else {
                    Button("Manage Models") {
                        selectedTab = 1
                    }
                    .buttonStyle(.glassProminent)
                }
            }
            .padding(28)
            .glassEffect(.clear, in: .rect(cornerRadius: 28))
            .padding()
            .frame(maxHeight: .infinity)

        case .ready:
            chatContent
        }
    }

    @ViewBuilder
    private var attachedDocuments: some View {
        if !viewModel.documents.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                GlassEffectContainer(spacing: 8) {
                    HStack(spacing: 8) {
                        ForEach(viewModel.documents) { document in
                            HStack(spacing: 7) {
                                documentStatusIcon(document.state)

                                Text(document.name)
                                    .font(.caption.weight(.medium))
                                    .lineLimit(1)

                                Button {
                                    viewModel.removeDocument(id: document.id)
                                } label: {
                                    Image(systemName: "xmark")
                                        .font(.caption2.bold())
                                }
                                .buttonStyle(.plain)
                                .accessibilityLabel("Remove \(document.name)")
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .glassEffect(
                                .regular.tint(Color.indigo.opacity(0.28)),
                                in: .capsule
                            )
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
            }
        }
    }

    @ViewBuilder
    private var chatContent: some View {
        if viewModel.messages.isEmpty {
            VStack(spacing: 18) {
                Image(systemName: "sparkles")
                    .font(.system(size: 32, weight: .semibold))
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 76, height: 76)
                    .glassEffect(.regular.tint(Color.indigo.opacity(0.35)), in: .circle)

                Text("Chat with Gemma")
                    .font(.title2.bold())

                Text(
                    viewModel.activeRoute == .phone
                        ? "Private, on-device inference powered by Cactus."
                        : "Private inference using the Gemma compute node on your Mac."
                )
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

                GlassEffectContainer(spacing: 10) {
                    HStack(spacing: 10) {
                        suggestionButton("Summarize an idea", icon: "text.alignleft")
                        suggestionButton("Plan a project", icon: "checklist")
                    }
                }
            }
            .padding(.horizontal, 24)
            .frame(maxHeight: .infinity)
        } else {
            ScrollViewReader { proxy in
                ScrollView {
                    GlassEffectContainer(spacing: 14) {
                        LazyVStack(spacing: 14) {
                            ForEach(viewModel.messages) { message in
                                messageBubble(message)
                                    .id(message.id)
                            }

                            if viewModel.isGenerating {
                                HStack(spacing: 10) {
                                    ProgressView()
                                    Text(
                                        isCreatingAdapter
                                            ? "Creating adapter file…"
                                            : "Gemma is thinking…"
                                    )
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                    Spacer()
                                }
                                .padding(.horizontal, 14)
                                .padding(.vertical, 12)
                                .glassEffect(.clear, in: .capsule)
                                .padding(.horizontal)
                            } else {
                                createAdapterButton
                            }
                        }
                        .padding(.vertical)
                    }
                }
                .scrollDismissesKeyboard(.interactively)
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
                Label("Listening… Tap the microphone to stop.", systemImage: "waveform")
                    .font(.caption)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 11)
                    .padding(.vertical, 7)
                    .glassEffect(.regular.tint(Color.red.opacity(0.45)), in: .capsule)
            } else if speechInput.isStarting {
                Label("Preparing on-device speech recognition…", systemImage: "waveform")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 11)
                    .padding(.vertical, 7)
                    .glassEffect(.clear, in: .capsule)
            } else if let errorMessage = speechInput.errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 11)
                    .padding(.vertical, 7)
                    .glassEffect(.regular.tint(Color.red.opacity(0.45)), in: .capsule)
            }

            GlassEffectContainer(spacing: 10) {
                HStack(alignment: .center, spacing: 10) {
                    Button {
                        isShowingDocumentPicker = true
                    } label: {
                        Image(systemName: "plus")
                            .fontWeight(.semibold)
                            .frame(width: 20, height: 20)
                    }
                    .buttonStyle(.glass)
                    .buttonBorderShape(.circle)
                    .controlSize(.large)
                    .glassEffectID("attachment", in: glassNamespace)
                    .disabled(
                        viewModel.isGenerating
                            || viewModel.hasPendingDocument
                            || speechInput.isListening
                            || speechInput.isStarting
                    )
                    .accessibilityLabel("Attach a document")

                    TextField("Ask anything", text: $prompt, axis: .vertical)
                        .textFieldStyle(.plain)
                        .lineLimit(1...5)
                        .focused($isPromptFocused)
                        .padding(.horizontal, 15)
                        .padding(.vertical, 12)
                        .frame(minHeight: 50, alignment: .center)
                        .glassEffect(.clear.interactive(), in: .rect(cornerRadius: 23))
                        .glassEffectID("messageField", in: glassNamespace)
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
                        .frame(width: 20, height: 20)
                    }
                    .buttonStyle(
                        .glass(
                            .regular.tint(
                                speechInput.isListening
                                    ? Color.red.opacity(0.65)
                                    : Color.cyan.opacity(0.25)
                            )
                        )
                    )
                    .buttonBorderShape(.circle)
                    .controlSize(.large)
                    .glassEffectID("microphone", in: glassNamespace)
                    .disabled(
                        viewModel.isGenerating
                            || viewModel.hasPendingDocument
                            || speechInput.isStarting
                    )
                    .accessibilityLabel(
                        speechInput.isListening ? "Stop listening" : "Start listening"
                    )

                    Button(action: send) {
                        Image(systemName: "arrow.up")
                            .fontWeight(.bold)
                            .frame(width: 20, height: 20)
                    }
                    .buttonStyle(.glassProminent)
                    .buttonBorderShape(.circle)
                    .controlSize(.large)
                    .tint(.indigo)
                    .glassEffectID("send", in: glassNamespace)
                    .disabled(
                        prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || !viewModel.canUseActiveRoute
                            || viewModel.isGenerating
                            || viewModel.hasPendingDocument
                            || viewModel.hasFailedDocument
                            || speechInput.isListening
                            || speechInput.isStarting
                    )
                    .accessibilityLabel("Send message")
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 12)
        .padding(.bottom, 12)
        .background {
            RoundedRectangle(cornerRadius: 32, style: .continuous)
                .fill(Color.black.opacity(0.94))
                .overlay {
                    RoundedRectangle(cornerRadius: 32, style: .continuous)
                        .stroke(Color.white.opacity(0.09), lineWidth: 1)
                }
                .shadow(color: Color.black.opacity(0.28), radius: 18, y: 8)
        }
        .padding(.horizontal, 8)
        .padding(.bottom, 6)
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
                        route.messageLabel,
                        systemImage: route.systemImage
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
            .padding(.vertical, 12)
            .foregroundStyle(.primary)
            .glassEffect(
                message.role == .user
                    ? .regular.tint(Color.indigo.opacity(0.5))
                    : .clear,
                in: .rect(cornerRadius: 22)
            )

            if message.role == .assistant {
                Spacer(minLength: 48)
            }
        }
        .padding(.horizontal)
    }

    private var createAdapterButton: some View {
        HStack {
            Spacer()
            Button(action: createAdapter) {
                Label("Create Adapter from Chat", systemImage: "wand.and.sparkles")
                    .font(.subheadline.weight(.semibold))
            }
            .buttonStyle(
                .glass(
                    .regular
                        .tint(Color.purple.opacity(0.35))
                        .interactive()
                )
            )
            .disabled(viewModel.isGenerating || isCreatingAdapter)
            Spacer()
        }
        .padding(.top, 4)
        .padding(.horizontal)
    }

    private func suggestionButton(_ title: String, icon: String) -> some View {
        Button {
            prompt = title
        } label: {
            Label(title, systemImage: icon)
                .font(.caption.weight(.semibold))
                .lineLimit(1)
        }
        .buttonStyle(.glass)
    }

    private var appBackground: some View {
        ZStack {
            Color(uiColor: .systemBackground)

            LinearGradient(
                colors: [
                    Color.indigo.opacity(0.28),
                    Color.purple.opacity(0.12),
                    Color.clear,
                    Color.cyan.opacity(0.08)
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            Circle()
                .fill(Color.blue.opacity(0.14))
                .frame(width: 330, height: 330)
                .blur(radius: 70)
                .offset(x: 150, y: -260)

            Circle()
                .fill(Color.purple.opacity(0.12))
                .frame(width: 280, height: 280)
                .blur(radius: 75)
                .offset(x: -170, y: 270)
        }
        .ignoresSafeArea()
    }

    private func adapterReadySheet(
        _ artifact: ConversationTrainingArtifact
    ) -> some View {
        NavigationStack {
            ZStack {
                appBackground

                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        HStack(spacing: 12) {
                            Image(systemName: "checkmark.seal.fill")
                                .font(.title2)
                                .foregroundStyle(.green)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Adapter File Ready")
                                    .font(.headline)
                                Text("Send it to your Mac when you’re ready to train.")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }

                        Text(artifact.contents)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                    }
                    .padding(18)
                    .glassEffect(.clear, in: .rect(cornerRadius: 26))
                    .padding()
                }
            }
            .navigationTitle("Create Adapter")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        adapterArtifact = nil
                    }
                }
            }
            .safeAreaInset(edge: .bottom) {
                ShareLink(item: artifact.url) {
                    Label("Share Adapter File", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.glassProminent)
                .tint(.indigo)
                .padding(.horizontal)
                .padding(.bottom, 8)
            }
        }
        .presentationDetents([.medium, .large])
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
        let shouldSpeakReply = speaksReplies && shouldSpeakNextReply
        prompt = ""
        shouldSpeakNextReply = false
        isPromptFocused = false
        speechOutput.stop()
        Task {
            if let response = await viewModel.send(submittedPrompt), shouldSpeakReply {
                speechOutput.speak(response)
            }
        }
    }

    private func createAdapter() {
        guard !isCreatingAdapter else { return }
        isCreatingAdapter = true
        adapterError = nil

        Task {
            do {
                adapterArtifact = try await viewModel.createTrainingSummary()
            } catch {
                adapterError = error.localizedDescription
            }
            isCreatingAdapter = false
        }
    }

    private func toggleListening() {
        if speechInput.isListening {
            Task {
                await speechInput.stopListening()
            }
        } else {
            isPromptFocused = false
            speechOutput.stop()
            shouldSpeakNextReply = false
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
