import CryptoKit
import Foundation

@MainActor
final class ModelManager: NSObject, ObservableObject {
    enum State: Equatable {
        case notDownloaded
        case downloading(progress: Double)
        case installing
        case installed(size: Int64)
        case failed(String)
    }

    static let archiveSize: Int64 = 2_721_189_793
    static let installedSize: Int64 = 4_027_519_369

    @Published private(set) var state: State = .notDownloaded

    private let archiveURL = URL(
        string: "https://huggingface.co/Cactus-Compute/gemma-4-E2B-it/resolve/main/gemma-4-e2b-it-cq4.zip?download=true"
    )!
    private let expectedSHA256 =
        "7a2579ad1089f626b15e121a1fa6a0d8294a4e7f1f7af38823620b5b5a8b8eec"
    private var downloadTask: URLSessionDownloadTask?
    private lazy var session: URLSession = {
        let configuration = URLSessionConfiguration.background(
            withIdentifier: "com.example.CactusApp.model-download"
        )
        configuration.allowsCellularAccess = true
        configuration.isDiscretionary = false
        configuration.sessionSendsLaunchEvents = true
        return URLSession(
            configuration: configuration,
            delegate: self,
            delegateQueue: nil
        )
    }()

    override init() {
        super.init()
        try? ModelStorage.migrateLegacyModelIfNeeded()
        refresh()
        Task {
            let tasks = await session.allTasks
            if let existing = tasks.compactMap({ $0 as? URLSessionDownloadTask }).first {
                downloadTask = existing
                state = .downloading(progress: max(existing.progress.fractionCompleted, 0))
            }
        }
    }

    func refresh() {
        if ModelStorage.isInstalled {
            state = .installed(size: Self.directorySize(at: ModelStorage.modelURL))
        } else if downloadTask == nil {
            state = .notDownloaded
        }
    }

    func download() {
        guard downloadTask == nil else { return }

        do {
            try FileManager.default.createDirectory(
                at: ModelStorage.modelsDirectory,
                withIntermediateDirectories: true
            )
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var directory = ModelStorage.modelsDirectory
            try directory.setResourceValues(values)
        } catch {
            state = .failed(error.localizedDescription)
            return
        }

        let task = session.downloadTask(with: archiveURL)
        downloadTask = task
        state = .downloading(progress: 0)
        task.resume()
    }

    func cancelDownload() {
        downloadTask?.cancel()
        downloadTask = nil
        state = .notDownloaded
    }

    func deleteModel() throws {
        downloadTask?.cancel()
        downloadTask = nil
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: ModelStorage.modelURL.path) {
            try fileManager.removeItem(at: ModelStorage.modelURL)
        }
        state = .notDownloaded
    }

    private func install(downloadedFile: URL) async {
        state = .installing
        let fileManager = FileManager.default
        let archiveDestination = ModelStorage.modelsDirectory
            .appendingPathComponent(".gemma-download.zip")
        let stagingDirectory = ModelStorage.modelsDirectory
            .appendingPathComponent(".gemma-installing", isDirectory: true)

        do {
            if fileManager.fileExists(atPath: archiveDestination.path) {
                try fileManager.removeItem(at: archiveDestination)
            }
            try fileManager.moveItem(at: downloadedFile, to: archiveDestination)

            let digest = try await Task.detached(priority: .utility) {
                try Self.sha256(at: archiveDestination)
            }.value
            guard digest == expectedSHA256 else {
                throw ModelManagementError.checksumMismatch
            }

            if fileManager.fileExists(atPath: stagingDirectory.path) {
                try fileManager.removeItem(at: stagingDirectory)
            }

            try await Task.detached(priority: .userInitiated) {
                try ArchiveExtractor.extractZip(
                    at: archiveDestination,
                    toDirectoryURL: stagingDirectory
                )
            }.value

            let extractedRoot = try Self.findBundleRoot(in: stagingDirectory)
            if fileManager.fileExists(atPath: ModelStorage.modelURL.path) {
                try fileManager.removeItem(at: ModelStorage.modelURL)
            }
            try fileManager.moveItem(at: extractedRoot, to: ModelStorage.modelURL)
            try fileManager.removeItem(at: stagingDirectory)
            try fileManager.removeItem(at: archiveDestination)

            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            var modelURL = ModelStorage.modelURL
            try modelURL.setResourceValues(values)
            state = .installed(size: Self.directorySize(at: ModelStorage.modelURL))
        } catch {
            try? fileManager.removeItem(at: archiveDestination)
            try? fileManager.removeItem(at: stagingDirectory)
            state = .failed(error.localizedDescription)
        }
    }

    nonisolated private static func sha256(at url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            guard let data = try handle.read(upToCount: 4 * 1_024 * 1_024),
                  !data.isEmpty else {
                break
            }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    nonisolated private static func findBundleRoot(in stagingDirectory: URL) throws -> URL {
        let fileManager = FileManager.default
        let directManifest = stagingDirectory.appendingPathComponent("components/manifest.json")
        if fileManager.fileExists(atPath: directManifest.path) {
            return stagingDirectory
        }

        let children = try fileManager.contentsOfDirectory(
            at: stagingDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        guard let root = children.first(where: {
            fileManager.fileExists(
                atPath: $0.appendingPathComponent("components/manifest.json").path
            )
        }) else {
            throw ModelManagementError.invalidBundle
        }
        return root
    }

    nonisolated private static func directorySize(at directory: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]
        ) else {
            return 0
        }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(
                forKeys: [.fileSizeKey, .isRegularFileKey]
            ), values.isRegularFile == true else {
                continue
            }
            total += Int64(values.fileSize ?? 0)
        }
        return total
    }
}

extension ModelManager: URLSessionDownloadDelegate {
    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let expected = totalBytesExpectedToWrite > 0
            ? totalBytesExpectedToWrite
            : 2_721_189_793
        let progress = min(max(Double(totalBytesWritten) / Double(expected), 0), 1)
        Task { @MainActor in
            state = .downloading(progress: progress)
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        let savedDownload = ModelStorage.modelsDirectory
            .appendingPathComponent(".gemma-background-download")
        do {
            try FileManager.default.createDirectory(
                at: ModelStorage.modelsDirectory,
                withIntermediateDirectories: true
            )
            if FileManager.default.fileExists(atPath: savedDownload.path) {
                try FileManager.default.removeItem(at: savedDownload)
            }
            // URLSession removes `location` when this delegate method returns.
            try FileManager.default.moveItem(at: location, to: savedDownload)
        } catch {
            Task { @MainActor in
                self.downloadTask = nil
                state = .failed(error.localizedDescription)
            }
            return
        }

        Task { @MainActor in
            self.downloadTask = nil
            await install(downloadedFile: savedDownload)
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard let error else { return }
        Task { @MainActor in
            downloadTask = nil
            let nsError = error as NSError
            if nsError.code == NSURLErrorCancelled {
                state = .notDownloaded
            } else {
                state = .failed(error.localizedDescription)
            }
        }
    }
}

private enum ModelManagementError: LocalizedError {
    case checksumMismatch
    case extractionFailed
    case invalidBundle

    var errorDescription: String? {
        switch self {
        case .checksumMismatch:
            "The downloaded model failed its security check. Please try again."
        case .extractionFailed:
            "The downloaded model could not be extracted."
        case .invalidBundle:
            "The download did not contain a valid Cactus model."
        }
    }
}
