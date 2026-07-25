import SwiftUI

enum ComputeNodeConfiguration {
    private static let serverURLKey = "computeNode.serverURL"
    private static let apiTokenKey = "computeNode.apiToken"

    static var serverAddress: String {
        get {
            UserDefaults.standard.string(forKey: serverURLKey)
                ?? "http://localhost:8080"
        }
        set {
            UserDefaults.standard.set(newValue, forKey: serverURLKey)
        }
    }

    static var apiToken: String {
        get {
            UserDefaults.standard.string(forKey: apiTokenKey)
                ?? "gemma-demo"
        }
        set {
            UserDefaults.standard.set(newValue, forKey: apiTokenKey)
        }
    }

    static var baseURL: URL? {
        let rawValue = serverAddress.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawValue.isEmpty else { return nil }
        let normalized = rawValue.hasSuffix("/")
            ? String(rawValue.dropLast())
            : rawValue
        return URL(string: normalized)
    }
}

@MainActor
final class ComputeNodeSettingsViewModel: ObservableObject {
    enum Status: Equatable {
        case idle
        case testing
        case connected(name: String, model: String)
        case failed(String)
    }

    @Published var serverAddress = ComputeNodeConfiguration.serverAddress
    @Published var apiToken = ComputeNodeConfiguration.apiToken
    @Published private(set) var status: Status = .idle

    private let client = ComputeNodeClient()

    func saveAndTest() async {
        ComputeNodeConfiguration.serverAddress =
            serverAddress.trimmingCharacters(in: .whitespacesAndNewlines)
        ComputeNodeConfiguration.apiToken =
            apiToken.trimmingCharacters(in: .whitespacesAndNewlines)
        status = .testing

        do {
            let health = try await client.health()
            guard health.inferenceReady else {
                status = .failed(
                    "The companion API is reachable, but its inference backend is not ready."
                )
                return
            }
            status = .connected(name: health.serverName, model: health.model)
        } catch {
            status = .failed(error.localizedDescription)
        }
    }
}

struct ComputeNodeSettingsView: View {
    @StateObject private var viewModel = ComputeNodeSettingsViewModel()

    var body: some View {
        NavigationStack {
            Form {
                Section("Companion server") {
                    TextField(
                        "http://192.168.1.10:8080",
                        text: $viewModel.serverAddress
                    )
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)

                    SecureField("API token", text: $viewModel.apiToken)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Button {
                        Task { await viewModel.saveAndTest() }
                    } label: {
                        if viewModel.status == .testing {
                            HStack {
                                ProgressView()
                                Text("Testing…")
                            }
                        } else {
                            Label("Save and test", systemImage: "network")
                        }
                    }
                    .disabled(viewModel.status == .testing)
                }

                Section("Status") {
                    statusContent
                }

                Section("Routing rule") {
                    Label(
                        "Messages without documents run privately on this iPhone.",
                        systemImage: "iphone"
                    )
                    Label(
                        "Attaching a document automatically requires the compute node.",
                        systemImage: "desktopcomputer"
                    )
                }

                Section {
                    Text(
                        "On a physical iPhone, localhost means the phone itself. Use the Mac’s local-network IP address."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Compute Node")
        }
    }

    @ViewBuilder
    private var statusContent: some View {
        switch viewModel.status {
        case .idle:
            Text("Not tested")
                .foregroundStyle(.secondary)
        case .testing:
            Text("Connecting…")
                .foregroundStyle(.secondary)
        case let .connected(name, model):
            Label("\(name) · \(model)", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case let .failed(message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
        }
    }
}
