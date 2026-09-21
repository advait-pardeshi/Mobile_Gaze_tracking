import Foundation
import simd
import QuartzCore
import CoreGraphics

/// Experiment 4 — prompted predictive communication task.
///
/// Where Experiment 3 shows a fixed 4×3 grid of word images and asks the
/// participant to compose one known sentence, this one asks a **question** and
/// rebuilds a 3×3 grid after every selection from a next-word predictor. The
/// participant answers by dwelling their way down the prediction tree.
///
/// **Why a coarser grid.** Nine cells at ~131×224 pt are as wide as
/// Experiment 3's cells (which work) and a third taller, so this asks strictly
/// less of the estimator than Experiment 3 does. The point is what that buys:
/// seven word slots reach ~7⁴ ≈ 2 400 four-word sentences, where a static grid
/// of nine cells reaches almost nothing. Prediction is what makes a grid coarse
/// enough for the tracker to resolve still expressive enough to talk with, and
/// this experiment is built to measure that trade rather than assert it.
///
/// **Layout.** Seven prediction slots, plus `⌫ back` (bottom-left) and
/// `✓ done` (bottom-right) at fixed positions for the whole run so they can be
/// learned motorically. The controls carry a *longer* dwell requirement: they
/// sit in the corners, where Experiment 2 reports the worst scatter, and an
/// accidental `⌫` destroys a word.
///
/// **Two-phase trial.** The question is spoken *and* shown while the grid is
/// hidden; then the text disappears and the grid goes live. Nothing competes
/// with the stimulus while gaze is being scored — which is what the codebase's
/// "no on-screen instructions" rule is actually protecting — and listening time
/// never enters the rate metric, because the response clock starts when the
/// utterance ends.
///
/// **Scoring.** Two conditions, see `Scoring`. Cued keeps `isCorrect` defined
/// exactly as in Experiment 3 and is the one that yields comparable numbers;
/// free is the realistic AAC task and is scored on time and corrections.
///
/// The metric this design exists for is the split between *the tracker picked
/// the wrong cell* and *the right word was never offered*: every grid state
/// records the word the answer needed next and where it ranked among the
/// candidates, so predictor failures and selection failures are separable in
/// post. Experiment 3 cannot distinguish those.

enum PredictiveCellKind: String {
    case word
    case back
    case done
    /// A word slot the predictor had no candidate for. Rendered, but inert:
    /// filling it would create selectable filler and score the predictor's
    /// shortfall as the participant's false selection.
    case blank
}

struct PredictiveCell: Identifiable {
    let index: Int
    let row: Int
    let col: Int
    let rect: CGRect
    let kind: PredictiveCellKind
    /// The word for `.word` cells; "" otherwise.
    let word: String

    var id: Int { index }
    var isSelectable: Bool { kind != .blank }
    /// Controls dwell longer than words — see the type comment.
    var isControl: Bool { kind == .back || kind == .done }
    var imageName: String { CommunicationWordSet.imageName(for: word) }
}

/// The grid as shown for one step of one trial, with what the answer needed
/// at that moment. One of these is recorded every time the grid changes.
struct PredictiveGridState {
    /// Run-global step counter, 0-based.
    let step: Int
    /// Index into the *shuffled* question order.
    let trialIndex: Int
    let questionKey: String
    /// Words chosen before this grid was built.
    let prefix: [String]
    /// Candidates the predictor returned, best-first.
    let candidates: [String]
    let cells: [PredictiveCell]
    /// Word the cued answer needed next, or "" in the free condition / once
    /// the answer is complete.
    let intendedWord: String
    /// Rank of `intendedWord` in `candidates` (0-based), or -1 when absent.
    /// -1 with a non-empty `intendedWord` is a **predictor** failure.
    let intendedRank: Int
    /// Seconds from run start at which this grid went up — the join key
    /// between `samples.csv` rows and the grid they were looking at.
    let atSeconds: Double
}

struct PredictiveSelection {
    /// Run-global selection order, 1-based.
    let order: Int
    let trialIndex: Int
    let questionKey: String
    let gridStep: Int
    let cellIndex: Int
    let kind: PredictiveCellKind
    let word: String
    /// Cued condition: was this the word the answer needed next? Always false
    /// for controls and in the free condition.
    let isCorrect: Bool
    /// Position in the cued answer this selection satisfied, or -1.
    let targetPosition: Int
    /// Rank of this word among the candidates shown (0-based), or -1.
    let chosenRank: Int
    /// What the answer needed at this step, and where it ranked.
    let intendedWord: String
    let intendedRank: Int
    /// Seconds from run start.
    let atSeconds: Double
    /// Seconds since the previous selection, or since the response phase
    /// began for the first selection of a trial.
    let sincePreviousSeconds: Double
    let predX: Double
    let predY: Double
}

struct PredictiveTrial {
    let trialIndex: Int
    let questionKey: String
    let questionText: String
    let cuedAnswer: [String]
    /// The answer as actually composed, controls excluded.
    let composed: [String]
    /// Cued: the composed answer equals the cued answer. Free: any non-empty
    /// answer the participant ended with `✓ done`.
    let completed: Bool
    /// Word selections (controls excluded).
    let wordSelections: Int
    let correctSelections: Int
    /// `⌫ back` presses — the error-recovery cost, and the free condition's
    /// error proxy.
    let corrections: Int
    /// Seconds the question was being presented.
    let promptSeconds: Double
    /// Seconds from the grid going live to the trial ending.
    let responseSeconds: Double
    /// Steps at which the needed word was not among the candidates.
    let predictorMisses: Int
    /// Steps at which a needed word existed at all (the denominator for the
    /// hit rate; 0 in the free condition).
    let predictorOpportunities: Int
    /// Selections the participant would have needed with no prediction, i.e.
    /// the length of the answer. Against `wordSelections` this is the
    /// keystroke cost of the predictor's mistakes.
    let idealSelections: Int
}

struct PredictiveTaskResult {
    enum Scoring: String {
        /// The participant is told which answer to give. `isCorrect` is
        /// defined as in Experiment 3, so accuracy and WPM are directly
        /// comparable to it.
        case cued
        /// The participant answers freely. No correctness; scored on
        /// time-to-answer, corrections and keystroke savings.
        case free
    }

    let scoring: Scoring
    let rows: Int
    let cols: Int
    let trials: [PredictiveTrial]
    let selections: [PredictiveSelection]
    let gridStates: [PredictiveGridState]
    let screenSize: CGSize
    let distancePoints: Double
    let durationSeconds: Double
    var runBundleURL: URL? = nil

    var completedTrials: Int { trials.filter(\.completed).count }
    var totalWordSelections: Int { trials.reduce(0) { $0 + $1.wordSelections } }
    var totalCorrect: Int { trials.reduce(0) { $0 + $1.correctSelections } }
    var totalCorrections: Int { trials.reduce(0) { $0 + $1.corrections } }

    /// Fraction of word selections that were the needed next word. Interface
    /// precision, not estimator precision. NaN in the free condition.
    var selectionAccuracy: Double {
        guard scoring == .cued, totalWordSelections > 0 else { return .nan }
        return Double(totalCorrect) / Double(totalWordSelections)
    }

    /// Fraction of steps at which the needed word was on the grid at all.
    /// This is the *predictor's* score, and the reason it is reported
    /// separately from `selectionAccuracy`: a low number here means the
    /// apparatus failed, not the participant.
    var predictorHitRate: Double {
        let opp = trials.reduce(0) { $0 + $1.predictorOpportunities }
        guard opp > 0 else { return .nan }
        let miss = trials.reduce(0) { $0 + $1.predictorMisses }
        return Double(opp - miss) / Double(opp)
    }

    /// Mean seconds per word actually committed, over completed trials —
    /// prompt time excluded, so this is the rate of the interface and not of
    /// the experimenter's script.
    var meanSecondsPerWord: Double {
        let done = trials.filter { $0.completed && $0.composed.count > 0 }
        guard !done.isEmpty else { return .nan }
        let secs = done.reduce(0.0) { $0 + $1.responseSeconds }
        let words = done.reduce(0) { $0 + $1.composed.count }
        guard words > 0 else { return .nan }
        return secs / Double(words)
    }

    var wordsPerMinute: Double {
        let s = meanSecondsPerWord
        guard s.isFinite, s > 0 else { return .nan }
        return 60.0 / s
    }

    /// Mean seconds to produce a whole answer, over completed trials.
    var meanSecondsPerAnswer: Double {
        let done = trials.filter(\.completed)
        guard !done.isEmpty else { return .nan }
        return done.reduce(0.0) { $0 + $1.responseSeconds } / Double(done.count)
    }

    /// Selections actually spent vs. the minimum the answer needed. 1.0 is
    /// perfect; 1.5 means half again as many dwells were spent recovering
    /// from wrong picks and missing candidates.
    var selectionOverhead: Double {
        let ideal = trials.reduce(0) { $0 + $1.idealSelections }
        guard ideal > 0 else { return .nan }
        let spent = trials.reduce(0) { $0 + $1.wordSelections + $1.corrections }
        return Double(spent) / Double(ideal)
    }

    static let masterLogFilename = "experiment4_predictive_log.csv"
    static func masterLogURL() -> URL {
        FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(masterLogFilename)
    }

    func csvString() -> String {
        var lines: [String] = []

        lines.append("# Selections")
        lines.append([
            "selection", "trial", "question", "grid_step", "cell_idx", "kind",
            "word", "correct", "target_pos", "chosen_rank",
            "intended_word", "intended_rank",
            "at_s", "since_prev_s", "pred_x", "pred_y"
        ].joined(separator: ","))
        for s in selections {
            var row: [String] = []
            row.append("\(s.order)")
            row.append("\(s.trialIndex)")
            row.append(s.questionKey)
            row.append("\(s.gridStep)")
            row.append("\(s.cellIndex)")
            row.append(s.kind.rawValue)
            row.append(s.word)
            row.append(s.isCorrect ? "1" : "0")
            row.append("\(s.targetPosition)")
            row.append("\(s.chosenRank)")
            row.append(s.intendedWord)
            row.append("\(s.intendedRank)")
            row.append(fmt(s.atSeconds, 3))
            row.append(fmt(s.sincePreviousSeconds, 3))
            row.append(fmt(s.predX, 2))
            row.append(fmt(s.predY, 2))
            lines.append(row.joined(separator: ","))
        }

        // Every grid the participant saw. `at_s` joins these to samples.csv:
        // a sample belongs to the last grid state whose at_s precedes it.
        lines.append("")
        lines.append("# Grid states")
        lines.append([
            "grid_step", "trial", "question", "at_s", "prefix",
            "candidates", "intended_word", "intended_rank"
        ].joined(separator: ","))
        for g in gridStates {
            var row: [String] = []
            row.append("\(g.step)")
            row.append("\(g.trialIndex)")
            row.append(g.questionKey)
            row.append(fmt(g.atSeconds, 3))
            row.append("\"\(g.prefix.joined(separator: " "))\"")
            row.append("\"\(g.candidates.joined(separator: " "))\"")
            row.append(g.intendedWord)
            row.append("\(g.intendedRank)")
            lines.append(row.joined(separator: ","))
        }

        // Cell geometry is constant across the run — only the payload
        // changes — so it is written once rather than per step.
        lines.append("")
        lines.append("# Cell geometry (constant for the run)")
        lines.append("cell_idx,row,col,kind_at_start,x_min,y_min,x_max,y_max")
        if let first = gridStates.first {
            for c in first.cells {
                var row: [String] = []
                row.append("\(c.index)")
                row.append("\(c.row)")
                row.append("\(c.col)")
                row.append(c.kind.rawValue)
                row.append(fmt(Double(c.rect.minX), 2))
                row.append(fmt(Double(c.rect.minY), 2))
                row.append(fmt(Double(c.rect.maxX), 2))
                row.append(fmt(Double(c.rect.maxY), 2))
                lines.append(row.joined(separator: ","))
            }
        }

        lines.append("")
        lines.append("# Trials")
        lines.append([
            "trial", "question", "question_text", "cued_answer", "composed",
            "completed", "word_selections", "correct", "corrections",
            "ideal_selections", "predictor_hits", "predictor_opportunities",
            "prompt_s", "response_s"
        ].joined(separator: ","))
        for t in trials {
            var row: [String] = []
            row.append("\(t.trialIndex)")
            row.append(t.questionKey)
            row.append("\"\(t.questionText)\"")
            row.append("\"\(t.cuedAnswer.joined(separator: " "))\"")
            row.append("\"\(t.composed.joined(separator: " "))\"")
            row.append(t.completed ? "1" : "0")
            row.append("\(t.wordSelections)")
            row.append("\(t.correctSelections)")
            row.append("\(t.corrections)")
            row.append("\(t.idealSelections)")
            row.append("\(t.predictorOpportunities - t.predictorMisses)")
            row.append("\(t.predictorOpportunities)")
            row.append(fmt(t.promptSeconds, 3))
            row.append(fmt(t.responseSeconds, 3))
            lines.append(row.joined(separator: ","))
        }

        lines.append("")
        lines.append("# Overall")
        lines.append("scoring,rows,cols,n_trials,completed_trials,word_selections,correct,corrections,selection_accuracy_pct,predictor_hit_rate_pct,selection_overhead,mean_s_per_word,mean_s_per_answer,words_per_min,duration_s,tz_points,screen_w_pt,screen_h_pt")
        var o: [String] = []
        o.append(scoring.rawValue)
        o.append("\(rows)")
        o.append("\(cols)")
        o.append("\(trials.count)")
        o.append("\(completedTrials)")
        o.append("\(totalWordSelections)")
        o.append("\(totalCorrect)")
        o.append("\(totalCorrections)")
        o.append(fmt(selectionAccuracy * 100.0, 2))
        o.append(fmt(predictorHitRate * 100.0, 2))
        o.append(fmt(selectionOverhead, 3))
        o.append(fmt(meanSecondsPerWord, 3))
        o.append(fmt(meanSecondsPerAnswer, 3))
        o.append(fmt(wordsPerMinute, 2))
        o.append(fmt(durationSeconds, 2))
        o.append(fmt(distancePoints, 2))
        o.append(fmt(Double(screenSize.width), 2))
        o.append(fmt(Double(screenSize.height), 2))
        lines.append(o.joined(separator: ","))
        return lines.joined(separator: "\n") + "\n"
    }

    private func fmt(_ x: Double, _ places: Int) -> String {
        guard x.isFinite else { return "" }
        return String(format: "%.\(places)f", x)
    }

    @discardableResult
    func appendToMasterLog() throws -> URL {
        let url = Self.masterLogURL()
        let runId = UserDefaults.standard
            .integer(forKey: "experiment4_next_run_id") + 1
        UserDefaults.standard.set(runId, forKey: "experiment4_next_run_id")
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let stamp = df.string(from: Date())
        var section = "\n=== Run \(runId) @ \(stamp) — Experiment 4 "
        section += "(predictive communication, \(scoring.rawValue)) ===\n"
        section += csvString()
        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            if let data = section.data(using: .utf8) {
                try handle.write(contentsOf: data)
            }
        } else {
            let first = section.hasPrefix("\n")
                ? String(section.dropFirst()) : section
            try first.write(to: url, atomically: true, encoding: .utf8)
        }
        return url
    }
}

@MainActor
final class PredictiveTaskController: ObservableObject {

    enum Phase: Equatable {
        case idle
        /// Question is being spoken/shown; the grid is hidden and no dwell
        /// accrues.
        case prompt
        /// Grid is live and scored.
        case responding
        case complete
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var cells: [PredictiveCell] = []
    /// Run-global grid counter. The overlay keys its transition off this so
    /// a refresh is visibly a refresh.
    @Published private(set) var gridStep: Int = -1
    /// Words chosen in the current trial, controls already applied.
    @Published private(set) var composed: [String] = []
    /// Index into the shuffled question order.
    @Published private(set) var trialIndex: Int = 0
    @Published private(set) var activeCellIndex: Int?
    @Published private(set) var dwellProgress: Double = 0
    @Published private(set) var lastSelectedCell: Int?
    /// True while the post-refresh lockout is holding off dwell.
    @Published private(set) var isLockedOut: Bool = false
    @Published private(set) var result: PredictiveTaskResult?

    let questions: [PromptQuestion]
    let scoring: PredictiveTaskResult.Scoring
    let screenSize: CGSize
    let calibration: CalibrationModel
    let rows = 3
    let cols = 3
    /// Grid indices that hold predicted words, best-first. Row 2 keeps its
    /// outer cells for the controls.
    static let wordSlots = [0, 1, 2, 3, 4, 5, 7]
    static let backSlot = 6
    static let doneSlot = 8

    let dwellRequirement: CFTimeInterval
    /// Longer than `dwellRequirement`: the controls sit in the corners, where
    /// scatter is worst, and `⌫` is destructive.
    let controlDwellRequirement: CFTimeInterval
    /// Dwell is suppressed for this long after every grid change. Without it
    /// an in-flight fixation keeps accruing onto whatever word landed in the
    /// cell, producing selections the participant never made.
    let refreshLockout: CFTimeInterval
    /// Floor on the prompt phase; it also ends only once speech has stopped.
    let promptMinSeconds: CFTimeInterval
    let trialTimeout: CFTimeInterval
    let maxSelectionsPerTrial: Int
    let sentenceBarHeight: CGFloat
    let controlBarHeight: CGFloat
    let gridArea: CGRect

    private let predictor: ResponsePredictor
    private let audio: WordAudioPlayer

    private var runStart: CFTimeInterval = 0
    private var promptStart: CFTimeInterval = 0
    private var responseStart: CFTimeInterval = 0
    private var lastSelectionTime: CFTimeInterval?
    private var lockoutUntil: CFTimeInterval = 0
    private var dwellCell: Int?
    private var dwellStart: CFTimeInterval = 0
    private var blockedCell: Int?
    private var timer: Timer?

    /// Per-trial accumulators, flushed into a `PredictiveTrial` at trial end.
    private var expectedIndex = 0
    private var composedCorrect: [Bool] = []
    private var trialWordSelections = 0
    private var trialCorrections = 0
    private var trialPredictorMisses = 0
    private var trialPredictorOpportunities = 0
    private var trialPromptSeconds: Double = 0
    /// False until the question has actually started being spoken. The
    /// previous trial's answer read-back is still in the synthesizer when a
    /// trial begins, and `speakQuestion` cancels whatever is speaking — so
    /// the prompt waits for the read-back rather than truncating it.
    private var promptSpoken = false

    private var trials: [PredictiveTrial] = []
    private var selections: [PredictiveSelection] = []
    private var gridStates: [PredictiveGridState] = []

    private var runLog: ExperimentRunLog
    private(set) var runBundleURL: URL?

    var onComplete: ((PredictiveTaskResult) -> Void)?

    init(screenSize: CGSize,
         calibration: CalibrationModel,
         audio: WordAudioPlayer,
         questions: [PromptQuestion] = PromptQuestionSet.session(),
         predictor: ResponsePredictor = ResponseTrie(),
         scoring: PredictiveTaskResult.Scoring = .cued,
         dwellRequirement: CFTimeInterval = 1.0,
         controlDwellRequirement: CFTimeInterval = 1.5,
         refreshLockout: CFTimeInterval = 0.4,
         promptMinSeconds: CFTimeInterval = 2.5,
         trialTimeout: CFTimeInterval = 60.0,
         maxSelectionsPerTrial: Int = 12,
         sentenceBarHeight: CGFloat = 96,
         controlBarHeight: CGFloat = 84) {
        self.screenSize = screenSize
        self.calibration = calibration
        self.audio = audio
        // Shuffled per run so a fatigue or practice effect can't ride on one
        // question's position in the sequence.
        self.questions = questions.shuffled()
        self.predictor = predictor
        self.scoring = scoring
        self.dwellRequirement = dwellRequirement
        self.controlDwellRequirement = controlDwellRequirement
        self.refreshLockout = refreshLockout
        self.promptMinSeconds = promptMinSeconds
        self.trialTimeout = trialTimeout
        self.maxSelectionsPerTrial = maxSelectionsPerTrial
        self.sentenceBarHeight = sentenceBarHeight
        self.controlBarHeight = controlBarHeight
        let h = max(0, screenSize.height - sentenceBarHeight - controlBarHeight)
        self.gridArea = CGRect(x: 0,
                               y: sentenceBarHeight,
                               width: screenSize.width,
                               height: h)
        self.runLog = ExperimentRunLog(
            experiment: "exp4",
            variant: "predictive_\(scoring.rawValue)",
            runLabel: "Experiment 4 (predictive communication, \(scoring.rawValue))",
            predictionSource: "smoothed",
            screenSize: screenSize,
            rows: 3,
            cols: 3,
            timing: ["dwell_requirement_s": dwellRequirement,
                     "control_dwell_requirement_s": controlDwellRequirement,
                     "refresh_lockout_s": refreshLockout,
                     "prompt_min_s": promptMinSeconds,
                     "trial_timeout_s": trialTimeout,
                     "n_questions": Double(questions.count)])
    }

    deinit { timer?.invalidate() }

    /// The question being answered right now.
    var currentQuestion: PromptQuestion? {
        trialIndex < questions.count ? questions[trialIndex] : nil
    }

    /// Word the cued answer needs next, or nil (free condition, or done).
    var expectedWord: String? {
        guard scoring == .cued, let q = currentQuestion else { return nil }
        return expectedIndex < q.cuedAnswer.count
            ? q.cuedAnswer[expectedIndex] : nil
    }

    /// Rect of a cell, for the overlay. Geometry is fixed for the run.
    func rect(row r: Int, col c: Int) -> CGRect {
        let w = gridArea.width / CGFloat(cols)
        let h = gridArea.height / CGFloat(rows)
        return CGRect(x: gridArea.minX + CGFloat(c) * w,
                      y: gridArea.minY + CGFloat(r) * h,
                      width: w,
                      height: h)
    }

    func start() {
        guard screenSize.width > 0, screenSize.height > 0 else {
            phase = .failed("No screen size")
            return
        }
        guard gridArea.width > 0, gridArea.height > 0 else {
            phase = .failed("Screen too small for the word grid")
            return
        }
        guard !questions.isEmpty else {
            phase = .failed("No questions")
            return
        }
        // Every trie must be closed: a prefix that does not finish the
        // sentence has to have an authored node, or the grid falls through to
        // the question's closers mid-response and offers words unrelated to
        // what was picked. That is an apparatus fault, not a hard trial — it
        // changes what the participant *can* say — so it fails the run rather
        // than quietly degrading it. Checked in both conditions: the free
        // condition is precisely where the off-path branches get walked.
        for q in questions {
            let dangling = q.danglingPrefixes(count: Self.wordSlots.count)
            if let first = dangling.first {
                phase = .failed("\(q.key): \(dangling.count) unauthored "
                                + "prefix(es), e.g. \"\(first)\"")
                return
            }
        }
        // In the cued condition every answer must be reachable through the
        // predictor, or the trial is unwinnable and would enter the aggregate
        // as a timeout indistinguishable from a participant who could not hit
        // the cells. This can only happen if the trie is misauthored, so fail
        // loudly rather than collect the run.
        if scoring == .cued {
            for q in questions {
                if let bad = q.unreachableCuedStep(count: Self.wordSlots.count) {
                    let w = q.cuedAnswer[bad]
                    phase = .failed("\(q.key): \"\(w)\" unreachable at step \(bad)")
                    return
                }
            }
        }
        timer?.invalidate()
        trials.removeAll()
        selections.removeAll()
        gridStates.removeAll()
        gridStep = -1
        trialIndex = 0
        runStart = CACurrentMediaTime()
        runLog.begin()
        let t = Timer(timeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
        RunLoop.main.add(t, forMode: .common)
        self.timer = t
        beginTrial()
    }

    func cancel() {
        timer?.invalidate()
        timer = nil
        audio.stop()
        phase = .idle
        activeCellIndex = nil
        dwellProgress = 0
        result = nil
    }

    /// End the run early, scoring whatever has been collected. The trial in
    /// progress is flushed as incomplete rather than discarded.
    func finishNow() {
        guard phase == .prompt || phase == .responding else { return }
        timer?.invalidate()
        timer = nil
        endTrial()
        finish()
    }

    // MARK: - Trial lifecycle

    private func beginTrial() {
        guard let q = currentQuestion else {
            finish()
            return
        }
        composed = []
        composedCorrect = []
        expectedIndex = 0
        trialWordSelections = 0
        trialCorrections = 0
        trialPredictorMisses = 0
        trialPredictorOpportunities = 0
        lastSelectionTime = nil
        resetDwell()
        blockedCell = nil
        lastSelectedCell = nil
        phase = .prompt
        promptSpoken = false
        promptStart = CACurrentMediaTime()
        // Built now so the grid is ready the instant the prompt ends; the
        // lockout is what actually gates scoring. The question itself is
        // spoken from `tick()`, once the previous answer has finished
        // playing back.
        refreshGrid()
    }

    private func beginResponse() {
        let now = CACurrentMediaTime()
        trialPromptSeconds = now - promptStart
        responseStart = now
        lastSelectionTime = nil
        lockoutUntil = now + refreshLockout
        isLockedOut = true
        phase = .responding
    }

    private func endTrial() {
        guard let q = currentQuestion else { return }
        let now = CACurrentMediaTime()
        let completed: Bool
        if scoring == .cued {
            completed = composed == q.cuedAnswer
        } else {
            completed = !composed.isEmpty
        }
        // Prompt time is excluded from `responseSeconds` on purpose: the rate
        // metric must not include how long the synthesizer took to speak.
        let elapsed = phase == .responding ? now - responseStart : 0
        let ideal = scoring == .cued ? q.cuedAnswer.count : composed.count
        trials.append(PredictiveTrial(
            trialIndex: trialIndex,
            questionKey: q.key,
            questionText: q.text,
            cuedAnswer: q.cuedAnswer,
            composed: composed,
            completed: completed,
            wordSelections: trialWordSelections,
            correctSelections: composedCorrect.filter { $0 }.count,
            corrections: trialCorrections,
            promptSeconds: trialPromptSeconds,
            responseSeconds: elapsed,
            predictorMisses: trialPredictorMisses,
            predictorOpportunities: trialPredictorOpportunities,
            idealSelections: ideal))
    }

    /// Close the current trial and move on, or finish the run.
    private func advanceTrial() {
        endTrial()
        // Speak the answer back before moving on — the participant needs to
        // hear what they actually produced, which is the feedback channel the
        // whole task is about.
        if !composed.isEmpty { audio.speakSentence(composed) }
        trialIndex += 1
        if trialIndex >= questions.count {
            timer?.invalidate()
            timer = nil
            finish()
        } else {
            beginTrial()
        }
    }

    // MARK: - Grid

    /// Rebuild the nine cells from the predictor for the current prefix.
    ///
    /// Rects never move; only the payload changes. Any dwell in flight is
    /// dropped and the lockout raised, so the fixation the participant was
    /// already holding cannot be charged to a word they have not yet seen.
    private func refreshGrid() {
        guard let q = currentQuestion else { return }
        let slots = Self.wordSlots
        let candidates = predictor.candidates(for: q,
                                              prefix: composed,
                                              count: slots.count)

        var built: [PredictiveCell] = []
        built.reserveCapacity(rows * cols)
        for idx in 0..<(rows * cols) {
            let r = idx / cols
            let c = idx % cols
            let rc = rect(row: r, col: c)
            let kind: PredictiveCellKind
            var word = ""
            if idx == Self.backSlot {
                kind = .back
            } else if idx == Self.doneSlot {
                kind = .done
            } else if let pos = slots.firstIndex(of: idx), pos < candidates.count {
                kind = .word
                word = candidates[pos]
            } else {
                kind = .blank
            }
            built.append(PredictiveCell(index: idx, row: r, col: c,
                                        rect: rc, kind: kind, word: word))
        }

        // What the answer needed here, and whether the predictor offered it.
        // Recorded per grid, not per selection, so a step the participant
        // never resolved still counts against the predictor.
        var intended = ""
        var intendedRank = -1
        if scoring == .cued, let want = expectedWord {
            intended = want
            intendedRank = candidates.firstIndex(of: want) ?? -1
            trialPredictorOpportunities += 1
            if intendedRank < 0 { trialPredictorMisses += 1 }
        }

        gridStep += 1
        cells = built
        gridStates.append(PredictiveGridState(
            step: gridStep,
            trialIndex: trialIndex,
            questionKey: q.key,
            prefix: composed,
            candidates: candidates,
            cells: built,
            intendedWord: intended,
            intendedRank: intendedRank,
            atSeconds: CACurrentMediaTime() - runStart))

        resetDwell()
        blockedCell = nil
        lockoutUntil = CACurrentMediaTime() + refreshLockout
        isLockedOut = true
    }

    // MARK: - Gaze

    func ingest(predictionScreenAbs: CGPoint,
                gazeCam: simd_double3,
                headPose: HeadPose,
                pupilDiameters: PupilMeasure.Diameters = .init(leftPx: .nan, rightPx: .nan),
                diagnostics: ExperimentRunLog.Diagnostics = .empty) {
        guard phase == .prompt || phase == .responding else { return }
        let now = CACurrentMediaTime()
        let inPrompt = phase == .prompt

        // The prompt phase is logged too: where the participant looks while
        // listening is the baseline the response is measured against, and it
        // timestamps the prompt window inside samples.csv.
        let hitCell = inPrompt
            ? nil
            : cells.first { $0.rect.contains(predictionScreenAbs) }

        // Hoisted into typed locals: inlining these into an 18-argument
        // initializer pushes Swift's expression type-checker into an
        // exponential blowup (same reason as CommunicationTaskController).
        let e = headPose.euler
        let logIdx: Int = hitCell?.index ?? -1
        let logRow: Int = hitCell?.row ?? -1
        let logCol: Int = hitCell?.col ?? -1
        let logRect: CGRect = hitCell?.rect ?? .null
        let logPhase: String = inPrompt ? "prompt" : "response"
        let order: Int = trialIndex + 1
        runLog.append(.init(
            t: now,
            trialOrder: order,
            cellIdx: logIdx,
            row: logRow,
            col: logCol,
            phase: logPhase,
            cellRect: logRect,
            predX: Double(predictionScreenAbs.x),
            predY: Double(predictionScreenAbs.y),
            headYawDeg: e.yaw,
            headPitchDeg: e.pitch,
            headRollDeg: e.roll,
            headTx: headPose.translation.x,
            headTy: headPose.translation.y,
            headTz: headPose.translation.z,
            pupilLeftPx: pupilDiameters.leftPx,
            pupilRightPx: pupilDiameters.rightPx,
            diag: diagnostics))

        guard !inPrompt else { return }

        // Samples inside the lockout are logged but never scored.
        if now < lockoutUntil {
            resetDwell()
            return
        }
        if isLockedOut { isLockedOut = false }

        guard let cell = hitCell, cell.isSelectable else {
            resetDwell()
            blockedCell = nil
            return
        }

        if let blocked = blockedCell, blocked != cell.index {
            blockedCell = nil
        }
        if blockedCell == cell.index {
            resetDwell()
            return
        }

        if dwellCell != cell.index {
            dwellCell = cell.index
            dwellStart = now
            activeCellIndex = cell.index
            dwellProgress = 0
            return
        }

        let need = cell.isControl ? controlDwellRequirement : dwellRequirement
        let held = now - dwellStart
        dwellProgress = max(0, min(1, held / need))
        if held >= need {
            select(cell, at: now, prediction: predictionScreenAbs)
        }
    }

    private func resetDwell() {
        if dwellCell != nil || activeCellIndex != nil {
            dwellCell = nil
            activeCellIndex = nil
            dwellProgress = 0
        }
    }

    private func select(_ cell: PredictiveCell,
                        at now: CFTimeInterval,
                        prediction: CGPoint) {
        guard let q = currentQuestion else { return }
        let state = gridStates.last
        let chosenRank = state?.candidates.firstIndex(of: cell.word) ?? -1
        let isCorrect = cell.kind == .word && cell.word == expectedWord

        selections.append(PredictiveSelection(
            order: selections.count + 1,
            trialIndex: trialIndex,
            questionKey: q.key,
            gridStep: gridStep,
            cellIndex: cell.index,
            kind: cell.kind,
            word: cell.word,
            isCorrect: isCorrect,
            targetPosition: isCorrect ? expectedIndex : -1,
            chosenRank: cell.kind == .word ? chosenRank : -1,
            intendedWord: state?.intendedWord ?? "",
            intendedRank: state?.intendedRank ?? -1,
            atSeconds: now - runStart,
            sincePreviousSeconds: now - (lastSelectionTime ?? responseStart),
            predX: Double(prediction.x),
            predY: Double(prediction.y)))
        lastSelectionTime = now
        lastSelectedCell = cell.index
        blockedCell = cell.index

        switch cell.kind {
        case .word:
            composed.append(cell.word)
            composedCorrect.append(isCorrect)
            trialWordSelections += 1
            audio.speak(cell.word)
            if isCorrect { expectedIndex += 1 }
        case .back:
            // Undo, including the correctness bookkeeping — otherwise a
            // deleted correct word would leave the cued answer permanently
            // one step ahead of what is on screen.
            if !composed.isEmpty {
                composed.removeLast()
                if composedCorrect.removeLast() { expectedIndex -= 1 }
            }
            trialCorrections += 1
            audio.speak("back")
        case .done:
            advanceTrial()
            return
        case .blank:
            return
        }

        // A cued answer that is complete ends the trial without requiring a
        // `✓ done` dwell, so the seconds-per-word figure stays comparable to
        // Experiment 3, which also ends on the last word of the sentence.
        if scoring == .cued, composed == q.cuedAnswer {
            advanceTrial()
            return
        }
        if trialWordSelections + trialCorrections >= maxSelectionsPerTrial {
            advanceTrial()
            return
        }
        refreshGrid()
    }

    private func tick() {
        let now = CACurrentMediaTime()
        switch phase {
        case .prompt:
            guard promptSpoken else {
                // Waiting on the previous trial's answer read-back.
                // The 0.3 s floor is not cosmetic: `AVSpeechSynthesizer`
                // reports `isSpeaking == false` for a beat after `speak()`
                // while it spins up, so checking it on the very next tick
                // would see the answer read-back as finished and talk over
                // it.
                if now - promptStart >= 0.3, !audio.isSpeaking,
                   let q = currentQuestion {
                    audio.speakQuestion(q.text)
                    promptSpoken = true
                    // Restarted here so `promptSeconds` measures the question
                    // presentation and not the read-back it waited on.
                    promptStart = now
                }
                return
            }
            // Ends on the later of the floor and the end of speech, so a
            // long question is never cut off and a short one still gets a
            // beat before the grid goes live.
            if now - promptStart >= promptMinSeconds, !audio.isSpeaking {
                beginResponse()
            }
        case .responding:
            if isLockedOut, now >= lockoutUntil { isLockedOut = false }
            if now - responseStart >= trialTimeout { advanceTrial() }
        default:
            break
        }
    }

    private func finish() {
        var r = PredictiveTaskResult(
            scoring: scoring,
            rows: rows,
            cols: cols,
            trials: trials,
            selections: selections,
            gridStates: gridStates,
            screenSize: screenSize,
            distancePoints: VisualAngle.distancePoints(for: calibration),
            durationSeconds: CACurrentMediaTime() - runStart)
        do {
            let out = try runLog.finish(
                trialsCSV: r.csvString(),
                summary: [
                    "scoring": scoring.rawValue,
                    "n_trials": trials.count,
                    "completed_trials": r.completedTrials,
                    "word_selections": r.totalWordSelections,
                    "correct": r.totalCorrect,
                    "corrections": r.totalCorrections,
                    "selection_accuracy_pct": r.selectionAccuracy * 100.0,
                    "predictor_hit_rate_pct": r.predictorHitRate * 100.0,
                    "selection_overhead": r.selectionOverhead,
                    "mean_s_per_word": r.meanSecondsPerWord,
                    "mean_s_per_answer": r.meanSecondsPerAnswer,
                    "words_per_min": r.wordsPerMinute,
                ])
            self.runBundleURL = out.zip
            r.runBundleURL = out.zip
            // Join this launch's session, so the operator can ship every run
            // of this experiment as one archive without hand-filtering.
            ExperimentSession.shared.record(experiment: "exp4",
                                            label: "predictive_\(scoring.rawValue)",
                                            directory: out.directory)
        } catch {
            print("experiment4 predictive run bundle failed: \(error)")
        }
        self.result = r
        self.activeCellIndex = nil
        self.dwellProgress = 0
        phase = .complete
        onComplete?(r)
    }
}
