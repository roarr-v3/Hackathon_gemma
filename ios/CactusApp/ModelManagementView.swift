import SwiftUI

struct ModelManagementView: View {
    @ObservedObject var modelManager: ModelManager
    let unloadModel: () async -> Void

    @State private var confirmDeletion = false
    @State private var deletionError: String?

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "brain.head.profile")
                                .font(.title2)
                                .foregroundStyle(.green)
                            VStack(alignment: .leading) {
                                Text("Gemma 4 E2B")
                                    .font(.headline)
                                Text("Cactus CQ4 · On-device")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            statusBadge
                        }

                        stateControls
                    }
                    .padding(.vertical, 6)
                } header: {
                    Text("Local Models")
                } footer: {
                    Text(
                        "Stored in On My iPhone › Cactus › Models. The model is excluded from iCloud Backup and remains available across app updates."
                    )
                }

                Section("Storage") {
                    LabeledContent("Download", value: format(ModelManager.archiveSize))
                    LabeledContent("Installed", value: format(ModelManager.installedSize))
                    LabeledContent("Temporary space needed", value: "About 7 GB")
                }
            }
            .navigationTitle("Models")
            .confirmationDialog(
                "Delete Gemma from this iPhone?",
                isPresented: $confirmDeletion,
                titleVisibility: .visible
            ) {
                Button("Delete Model", role: .destructive) {
                    Task {
                        await unloadModel()
                        do {
                            try modelManager.deleteModel()
                        } catch {
                            deletionError = error.localizedDescription
                        }
                    }
                }
            } message: {
                Text("You can download it again later.")
            }
            .alert(
                "Couldn’t Delete Model",
                isPresented: Binding(
                    get: { deletionError != nil },
                    set: { if !$0 { deletionError = nil } }
                )
            ) {
                Button("OK") { deletionError = nil }
            } message: {
                Text(deletionError ?? "")
            }
        }
    }

    @ViewBuilder
    private var stateControls: some View {
        switch modelManager.state {
        case .notDownloaded:
            Button("Download Model", systemImage: "arrow.down.circle") {
                modelManager.download()
            }
            .buttonStyle(.borderedProminent)

        case let .downloading(progress):
            VStack(alignment: .leading, spacing: 8) {
                ProgressView(value: progress)
                HStack {
                    Text("\(Int(progress * 100))% downloaded")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Cancel", role: .destructive) {
                        modelManager.cancelDownload()
                    }
                    .font(.caption)
                }
            }

        case .installing:
            HStack {
                ProgressView()
                Text("Verifying and installing…")
                    .foregroundStyle(.secondary)
            }

        case let .installed(size):
            HStack {
                Label("\(format(size)) on this iPhone", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Spacer()
                Button("Delete", role: .destructive) {
                    confirmDeletion = true
                }
            }
        case let .failed(message):
            VStack(alignment: .leading, spacing: 8) {
                Label(message, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                Button("Try Again") {
                    modelManager.download()
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    @ViewBuilder
    private var statusBadge: some View {
        switch modelManager.state {
        case .installed:
            Text("Installed")
                .foregroundStyle(.green)
        case .downloading:
            Text("Downloading")
                .foregroundStyle(.blue)
        case .installing:
            Text("Installing")
                .foregroundStyle(.orange)
        case .failed:
            Text("Error")
                .foregroundStyle(.red)
        case .notDownloaded:
            Text("Not installed")
                .foregroundStyle(.secondary)
        }
    }

    private func format(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
