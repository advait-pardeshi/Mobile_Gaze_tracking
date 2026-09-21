import Foundation
import AVFoundation

/// Speaks a word aloud when it is selected in Experiment 3, and the prompted
/// question in Experiment 4.
///
/// Uses `AVSpeechSynthesizer` rather than bundled recordings so the word list
/// can change without shipping new audio assets.
///
/// Two behaviours matter for the experiment rather than for the audio:
///
///  * **Selections are never queued.** `AVSpeechSynthesizer` enqueues
///    utterances by default, so a participant selecting several words in quick
///    succession would hear the audio drift further and further behind the
///    screen. Each new word stops the previous one immediately, keeping the
///    feedback aligned with what was just selected.
///  * **The session category is `.playback` with `.mixWithOthers`.** The app
///    holds an active capture session for the front camera; taking an
///    exclusive audio session would risk interrupting it mid-run.
@MainActor
final class WordAudioPlayer {

    private let synthesizer = AVSpeechSynthesizer()
    /// Set once, lazily — configuring the session on every utterance is a
    /// measurable hitch on the main thread, which is also the gaze path.
    private var sessionConfigured = false

    /// Slightly slower than the default so a single short word is
    /// intelligible; the default rate clips one-syllable words.
    var rate: Float = 0.45
    var language: String = "en-US"

    /// Spoken-form overrides, keyed by the lowercased word.
    ///
    /// A bare "I" is read by `AVSpeechSynthesizer` as a *letter*, which the
    /// voice announces as "capital I" — the participant hears the case of the
    /// glyph instead of the word they selected. Spelling it phonetically makes
    /// the synthesizer treat it as a word again; what comes out is "I".
    private static let spokenOverrides: [String: String] = ["i": "eye"]

    /// The text actually handed to the synthesizer: every word replaced by its
    /// spoken form, if it has one. Applied to every utterance so a word is
    /// pronounced the same on selection, on clear, and in sentence playback.
    private static func spoken(_ text: String) -> String {
        text.split(separator: " ").map { token -> String in
            let key = String(token).lowercased()
            return spokenOverrides[key] ?? String(token)
        }.joined(separator: " ")
    }

    /// True while an utterance is in flight. Experiment 4 polls this to end
    /// its prompt phase only once the question has actually finished being
    /// spoken, so listening time never lands in the response clock.
    var isSpeaking: Bool { synthesizer.isSpeaking }

    /// Speak a prompted question (Experiment 4).
    ///
    /// Slightly faster than `rate`: a whole sentence read at the
    /// single-word rate is unnaturally slow, and the prompt phase is dead
    /// time in the run.
    func speakQuestion(_ text: String) {
        let q = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }
        configureSessionIfNeeded()
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        let utterance = AVSpeechUtterance(string: Self.spoken(q))
        utterance.rate = min(0.5, rate + 0.05)
        utterance.voice = AVSpeechSynthesisVoice(language: language)
        synthesizer.speak(utterance)
    }

    /// Speak `word`, cancelling anything currently being spoken.
    func speak(_ word: String) {
        let text = word.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        configureSessionIfNeeded()
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        let utterance = AVSpeechUtterance(string: Self.spoken(text))
        utterance.rate = rate
        utterance.voice = AVSpeechSynthesisVoice(language: language)
        synthesizer.speak(utterance)
    }

    /// Speak a whole sentence — used on the results screen to play back the
    /// composed sentence.
    func speakSentence(_ words: [String]) {
        let sentence = words.joined(separator: " ")
        guard !sentence.isEmpty else { return }
        configureSessionIfNeeded()
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        let utterance = AVSpeechUtterance(string: Self.spoken(sentence))
        utterance.rate = rate
        utterance.voice = AVSpeechSynthesisVoice(language: language)
        synthesizer.speak(utterance)
    }

    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
    }

    private func configureSessionIfNeeded() {
        guard !sessionConfigured else { return }
        sessionConfigured = true
        do {
            try AVAudioSession.sharedInstance().setCategory(
                .playback, mode: .spokenAudio, options: [.mixWithOthers])
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            // Non-fatal: speech may still play, and the experiment's
            // measurements don't depend on audio succeeding.
            print("[Audio] session setup failed: \(error)")
        }
    }
}
