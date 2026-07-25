import AVFoundation
import Speech

enum VoiceInteractionError: LocalizedError {
    case microphonePermissionDenied
    case speechPermissionDenied
    case speechRecognitionUnavailable
    case unsupportedLocale
    case microphoneUnavailable
    case invalidAudioFormat

    var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Microphone access is required to capture your voice."
        case .speechPermissionDenied:
            return "Speech Recognition access is required to transcribe on this iPhone."
        case .speechRecognitionUnavailable:
            return "Speech recognition is temporarily unavailable."
        case .unsupportedLocale:
            return "The current language is not supported for on-device transcription."
        case .microphoneUnavailable:
            return "No microphone is available for speech input."
        case .invalidAudioFormat:
            return "The microphone did not provide a usable audio format."
        }
    }
}

@MainActor
final class SpeechInputService: ObservableObject {
    @Published private(set) var transcript = ""
    @Published private(set) var isListening = false
    @Published private(set) var isStarting = false
    @Published private(set) var errorMessage: String?

    private var session: SpeechAnalyzerSession?

    func startListening() async {
        guard !isListening, !isStarting else { return }

        isStarting = true
        errorMessage = nil
        transcript = ""

        do {
            try await requestPermissions()
            guard SpeechTranscriber.isAvailable else {
                throw VoiceInteractionError.speechRecognitionUnavailable
            }

            let nextSession = SpeechAnalyzerSession(
                onTranscript: { [weak self] text in
                    self?.transcript = text
                },
                onError: { [weak self] error in
                    self?.errorMessage = error.localizedDescription
                }
            )

            session = nextSession
            try await nextSession.start()
            isListening = true
        } catch {
            session = nil
            errorMessage = error.localizedDescription
        }

        isStarting = false
    }

    func stopListening() async {
        guard let session else {
            isListening = false
            isStarting = false
            return
        }

        isListening = false
        await session.stop()
        self.session = nil
    }

    func clearError() {
        errorMessage = nil
    }

    private func requestPermissions() async throws {
        let speechAuthorization: SFSpeechRecognizerAuthorizationStatus
        switch SFSpeechRecognizer.authorizationStatus() {
        case .notDetermined:
            speechAuthorization = await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { status in
                    continuation.resume(returning: status)
                }
            }
        case let status:
            speechAuthorization = status
        }

        guard speechAuthorization == .authorized else {
            throw VoiceInteractionError.speechPermissionDenied
        }

        let microphoneGranted = await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }

        guard microphoneGranted else {
            throw VoiceInteractionError.microphonePermissionDenied
        }
    }
}

@MainActor
private final class SpeechAnalyzerSession {
    private let onTranscript: @MainActor (String) -> Void
    private let onError: @MainActor (Error) -> Void

    private var analyzer: SpeechAnalyzer?
    private var captureProvider: CaptureInputSequenceProvider?
    private var analyzerTask: Task<Void, Never>?
    private var resultsTask: Task<Void, Never>?
    private var finalizedTranscript = ""
    private var volatileTranscript = ""

    init(
        onTranscript: @escaping @MainActor (String) -> Void,
        onError: @escaping @MainActor (Error) -> Void
    ) {
        self.onTranscript = onTranscript
        self.onError = onError
    }

    func start() async throws {
        guard
            let supportedLocale = await SpeechTranscriber.supportedLocale(
                equivalentTo: Locale.current
            )
        else {
            throw VoiceInteractionError.unsupportedLocale
        }

        let transcriber = SpeechTranscriber(
            locale: supportedLocale,
            preset: .progressiveTranscription
        )
        try await installAssetsIfNeeded(for: transcriber)

        guard let microphone = AVCaptureDevice.default(for: .audio) else {
            throw VoiceInteractionError.microphoneUnavailable
        }

        let captureProvider = try await CaptureInputSequenceProvider.providerWithSession(
            from: microphone,
            compatibleWith: [transcriber],
            priority: .userInitiated
        )
        let analyzer = SpeechAnalyzer(modules: [transcriber])

        self.analyzer = analyzer
        self.captureProvider = captureProvider

        let analyzerFormat = await SpeechAnalyzer.bestAvailableAudioFormat(
            compatibleWith: [transcriber]
        )
        try await analyzer.prepareToAnalyze(in: analyzerFormat)

        resultsTask = Task { [weak self] in
            do {
                for try await result in transcriber.results {
                    guard let self else { return }
                    let text = String(result.text.characters)

                    if result.isFinal {
                        finalizedTranscript += text
                        volatileTranscript = ""
                    } else {
                        volatileTranscript = text
                    }

                    onTranscript(finalizedTranscript + volatileTranscript)
                }
            } catch is CancellationError {
                return
            } catch {
                self?.onError(error)
            }
        }

        analyzerTask = Task { [weak self] in
            do {
                try await analyzer.start(inputSequence: captureProvider.analyzerInputs)
            } catch is CancellationError {
                return
            } catch {
                self?.onError(error)
            }
        }

        captureProvider.captureSession.startRunning()
    }

    func stop() async {
        if captureProvider?.captureSession.isRunning == true {
            captureProvider?.captureSession.stopRunning()
        }

        do {
            try await analyzer?.finalize(through: nil)
        } catch {
            onError(error)
        }
        await analyzer?.cancelAndFinishNow()

        analyzerTask?.cancel()
        resultsTask?.cancel()
        analyzerTask = nil
        resultsTask = nil
        analyzer = nil
        captureProvider = nil
    }

    private func installAssetsIfNeeded(for transcriber: SpeechTranscriber) async throws {
        let modules: [any SpeechModule] = [transcriber]
        let status = await AssetInventory.status(forModules: modules)

        switch status {
        case .installed:
            return
        case .supported, .downloading:
            if let request = try await AssetInventory.assetInstallationRequest(
                supporting: modules
            ) {
                try await request.downloadAndInstall()
            }
        case .unsupported:
            throw VoiceInteractionError.unsupportedLocale
        @unknown default:
            throw VoiceInteractionError.speechRecognitionUnavailable
        }
    }
}

@MainActor
final class SpeechOutputService: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published private(set) var isSpeaking = false

    private let synthesizer = AVSpeechSynthesizer()

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func speak(_ text: String, language: String = Locale.current.identifier) {
        let cleanedText = text.replacingOccurrences(
            of: #"\[Source\s+\d+\]"#,
            with: "",
            options: .regularExpression
        )
        guard !cleanedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }

        stop()

        let utterance = AVSpeechUtterance(string: cleanedText)
        utterance.voice = preferredVoice(for: language)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1
        utterance.volume = 1
        synthesizer.speak(utterance)
        isSpeaking = true
    }

    func stop() {
        guard synthesizer.isSpeaking || synthesizer.isPaused else {
            isSpeaking = false
            return
        }
        synthesizer.stopSpeaking(at: .immediate)
        isSpeaking = false
    }

    private func preferredVoice(for language: String) -> AVSpeechSynthesisVoice? {
        let languageCode = Locale(identifier: language).language.languageCode?.identifier
        let candidates = AVSpeechSynthesisVoice.speechVoices().filter { voice in
            guard let languageCode else { return voice.language == language }
            return Locale(identifier: voice.language).language.languageCode?.identifier
                == languageCode
        }

        return candidates.max { left, right in
            left.quality.rawValue < right.quality.rawValue
        } ?? AVSpeechSynthesisVoice(language: language)
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didFinish utterance: AVSpeechUtterance
    ) {
        Task { @MainActor [weak self] in
            self?.isSpeaking = false
        }
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didCancel utterance: AVSpeechUtterance
    ) {
        Task { @MainActor [weak self] in
            self?.isSpeaking = false
        }
    }
}
