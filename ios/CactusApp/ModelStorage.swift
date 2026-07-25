import Foundation

enum ModelStorage {
    static let folderName = "gemma-4-e2b-it-cq4"

    static var modelsDirectory: URL {
        let documents = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        )[0]
        return documents.appendingPathComponent("Models", isDirectory: true)
    }

    static var modelURL: URL {
        modelsDirectory.appendingPathComponent(folderName, isDirectory: true)
    }

    static var manifestURL: URL {
        modelURL.appendingPathComponent("components/manifest.json")
    }

    static var isInstalled: Bool {
        FileManager.default.fileExists(atPath: manifestURL.path)
    }

    static func migrateLegacyModelIfNeeded() throws {
        guard !isInstalled else { return }
        let fileManager = FileManager.default
        guard let applicationSupport = fileManager.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            return
        }

        let legacyModels = applicationSupport
            .appendingPathComponent("CactusModels", isDirectory: true)
        let legacyModel = legacyModels
            .appendingPathComponent(folderName, isDirectory: true)
        let legacyManifest = legacyModel
            .appendingPathComponent("components/manifest.json")
        guard fileManager.fileExists(atPath: legacyManifest.path) else { return }

        try fileManager.createDirectory(
            at: modelsDirectory,
            withIntermediateDirectories: true
        )
        try fileManager.moveItem(at: legacyModel, to: modelURL)

        if let remaining = try? fileManager.contentsOfDirectory(atPath: legacyModels.path),
           remaining.isEmpty {
            try? fileManager.removeItem(at: legacyModels)
        }

        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        var mutableModelURL = modelURL
        try mutableModelURL.setResourceValues(values)
    }
}
