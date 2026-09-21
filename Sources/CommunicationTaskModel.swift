import Foundation
import simd
import QuartzCore
import CoreGraphics

/// Experiment 3 — image-based communication task.
///
/// Stimulus: a grid of word images. Five of them spell the target sentence
/// ("I want to drink water"); the rest are distractors. The participant
/// composes the sentence by fixating each word in order, and every selection
/// is spoken aloud.
///
/// This is the applied end of the pipeline. Experiments 1 and 2 measure the
/// estimator against geometry; this one measures whether the estimator is good
/// enough to *drive an AAC keyboard*, which is a different and harder question
/// — it depends on selection latency and false-selection rate, not just on
/// mean error.
///
/// **Selection model.** A word is selected when the gaze holds its cell
/// continuously for `dwellRequirement` seconds. Every selection is appended to
/// the composed sentence, right or wrong, because that is what a real AAC
/// device does — suppressing wrong selections would hide the error rate this
/// experiment exists to measure. After a selection the same cell cannot be
/// re-selected until the gaze has left it, so one long fixation produces one
/// word rather than a stream of repeats.
///
/// The run has **no automatic end**: it stops only when the operator taps
/// Finish (score and log it) or Cancel (discard it). There is deliberately no
/// selection cap and no run timeout — a participant who has finished the
/// target sentence can keep composing, and the operator decides when there is
/// enough data. `completed` in the result still reports whether the target
/// sentence itself was reproduced in order.

/// The words shown, and which of them form the target sentence.
struct CommunicationWordSet {
    /// Ordered target sentence.
    let target: [String]
    /// Distractors filling the rest of the grid.
    let distractors: [String]
    let rows: Int
    let cols: Int

    /// Default set: "I want to drink water" plus 6 distractors and the Clear
    /// tile, filling a 4×3 grid.
    ///
    /// Distractors are deliberately plausible AAC vocabulary rather than
    /// nonsense — a false selection only costs something if the wrong word was
    /// a realistic competitor.
    static let standard = CommunicationWordSet(
        target: ["I", "want", "to", "drink", "water"],
        distractors: ["eat", "sleep", "more", "please", "stop", "home"],
        rows: 4,
        cols: 3)

    /// The Clear control lives *on the grid* as a full-size tile rather than in
    /// the bottom bar, so the participant can trigger it by dwell like any
    /// other cell instead of needing the operator to tap for them.
    static let clearWord = "clear"

    /// Words competing for the shuffled cells. Clear is not among them: it is
    /// pinned to `clearCellIndex`.
    var allWords: [String] { target + distractors }

    /// Bottom-left cell. Clear is the one tile that does *not* move between
    /// runs — a control the participant has to hunt for is a control they
    /// stop using, and its position is not what the task is measuring.
    var clearCellIndex: Int { (rows - 1) * cols }

    /// Resource basename convention for the bundled word PNGs:
    /// `word_<lowercased>.png`. Falls back to a rendered text tile in the
    /// overlay when the asset is missing.
    static func imageName(for word: String) -> String {
        "word_" + word.lowercased()
            .replacingOccurrences(of: " ", with: "_")
    }
}

/// One cell of the word grid.
struct CommunicationCell: Identifiable {
    let index: Int
    let row: Int
    let col: Int
    let word: String
    /// Absolute screen points.
    let rect: CGRect
    /// True if this word appears in the target sentence.
    let isTargetWord: Bool

    var id: Int { index }
    var imageName: String { CommunicationWordSet.imageName(for: word) }
    /// The Clear action tile rather than a vocabulary word.
    var isClear: Bool { word == CommunicationWordSet.clearWord }
}

/// One selection event.
struct CommunicationSelection {
    /// 1-based selection order.
    let order: Int
    let cellIndex: Int
    let word: String
    /// Was this the word the target sentence needed next?
    let isCorrect: Bool
    /// Position in the target sentence this selection satisfied, or -1.
    let targetPosition: Int
    /// Seconds from run start to the moment of selection.
    let atSeconds: Double
    /// Seconds since the previous selection (or run start for the first).
    let sincePreviousSeconds: Double
    /// Prediction at the moment of selection, absolute screen points.
    let predX: Double
    let predY: Double
}

/// One Clear selection — a backspace that removes the last word from the
/// composed sentence.
///
/// A clear retracts the word from the *strip*, not from the record: the
/// selection stays in `selections`, because a wrong pick the participant then
/// cleared still happened and still counts against the interface. The clear is
/// logged so the analysis can separate corrected errors from uncorrected ones.
struct CommunicationClear {
    /// Number of selections that had been made when Clear fired.
    let afterSelection: Int
    /// The word taken off the strip.
    let removedWord: String
    /// Was that word a correct one, i.e. did the clear rewind the sentence?
    let removedWasCorrect: Bool
    /// Seconds from run start to the clear.
    let atSeconds: Double
}

struct CommunicationTaskResult {
    let wordSet: CommunicationWordSet
    let cells: [CommunicationCell]
    let selections: [CommunicationSelection]
    /// Clears tapped during the run, in order.
    let clears: [CommunicationClear]
    let screenSize: CGSize
    let distancePoints: Double
    let durationSeconds: Double
    /// True if the whole target sentence was composed in order.
    let completed: Bool
    var runBundleURL: URL? = nil

    /// Every word selected, in order — the sentence as actually composed.
    var composedSentence: [String] { selections.map(\.word) }

    var clearCount: Int { clears.count }

    var correctCount: Int { selections.filter(\.isCorrect).count }
    var incorrectCount: Int { selections.count - correctCount }

    /// Fraction of selections that were the needed next word. This is the
    /// selection precision of the interface, not of the gaze estimator.
    var selectionAccuracy: Double {
        selections.isEmpty
            ? .nan
            : Double(correctCount) / Double(selections.count)
    }

    /// Mean seconds between consecutive selections — the interface's
    /// effective words-per-minute driver.
    var meanSecondsPerSelection: Double {
        let gaps = selections.map(\.sincePreviousSeconds).filter { $0.isFinite }
        guard !gaps.isEmpty else { return .nan }
        return gaps.reduce(0, +) / Double(gaps.count)
    }

    /// Mean seconds per *correct* word — the figure that matters for how fast
    /// a real sentence can actually be produced.
    var meanSecondsPerCorrectWord: Double {
        guard correctCount > 0 else { return .nan }
        guard let last = selections.last(where: \.isCorrect) else { return .nan }
        return last.atSeconds / Double(correctCount)
    }

    /// Effective communication rate in words per minute, counting only the
    /// correct words that advanced the sentence.
    var wordsPerMinute: Double {
        let s = meanSecondsPerCorrectWord
        guard s.isFinite, s > 0 else { return .nan }
        return 60.0 / s
    }

    static let masterLogFilename = "experiment3_communication_log.csv"
    static func masterLogURL() -> URL {
        FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(masterLogFilename)
    }

    func csvString() -> String {
        var lines: [String] = []
        lines.append([
            "selection", "cell_idx", "word", "correct", "target_pos",
            "at_s", "since_prev_s", "pred_x", "pred_y"
        ].joined(separator: ","))
        for s in selections {
            var row: [String] = []
            row.append("\(s.order)")
            row.append("\(s.cellIndex)")
            row.append(s.word)
            row.append(s.isCorrect ? "1" : "0")
            row.append("\(s.targetPosition)")
            row.append(fmt(s.atSeconds, 3))
            row.append(fmt(s.sincePreviousSeconds, 3))
            row.append(fmt(s.predX, 2))
            row.append(fmt(s.predY, 2))
            lines.append(row.joined(separator: ","))
        }

        lines.append("")
        lines.append("# Grid layout")
        lines.append("cell_idx,row,col,word,is_target_word,x_min,y_min,x_max,y_max")
        for c in cells {
            var row: [String] = []
            row.append("\(c.index)")
            row.append("\(c.row)")
            row.append("\(c.col)")
            row.append(c.word)
            row.append(c.isTargetWord ? "1" : "0")
            row.append(fmt(Double(c.rect.minX), 2))
            row.append(fmt(Double(c.rect.minY), 2))
            row.append(fmt(Double(c.rect.maxX), 2))
            row.append(fmt(Double(c.rect.maxY), 2))
            lines.append(row.joined(separator: ","))
        }

        lines.append("")
        lines.append("# Clears")
        lines.append("clear_idx,after_selection,removed_word,removed_was_correct,at_s")
        for (i, c) in clears.enumerated() {
            lines.append("\(i + 1),\(c.afterSelection),\(c.removedWord),"
                         + "\(c.removedWasCorrect ? 1 : 0),\(fmt(c.atSeconds, 3))")
        }

        lines.append("")
        lines.append("# Overall")
        lines.append("completed,target_sentence,composed_sentence,n_selections,n_clears,correct,incorrect,selection_accuracy_pct,mean_s_per_selection,mean_s_per_correct_word,words_per_min,duration_s,tz_points,screen_w_pt,screen_h_pt")
        var o: [String] = []
        o.append(completed ? "1" : "0")
        // Quoted: the sentences contain spaces, and a bare space would still
        // parse but reads as a broken field in most viewers.
        o.append("\"\(wordSet.target.joined(separator: " "))\"")
        o.append("\"\(composedSentence.joined(separator: " "))\"")
        o.append("\(selections.count)")
        o.append("\(clearCount)")
        o.append("\(correctCount)")
        o.append("\(incorrectCount)")
        o.append(fmt(selectionAccuracy * 100.0, 2))
        o.append(fmt(meanSecondsPerSelection, 3))
        o.append(fmt(meanSecondsPerCorrectWord, 3))
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
            .integer(forKey: "experiment3_next_run_id") + 1
        UserDefaults.standard.set(runId, forKey: "experiment3_next_run_id")
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let stamp = df.string(from: Date())
        var section = "\n=== Run \(runId) @ \(stamp) — Experiment 3 (communication task) ===\n"
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
final class CommunicationTaskController: ObservableObject {

    enum Phase: Equatable {
        case idle
        case running
        case complete
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var cells: [CommunicationCell]
    @Published private(set) var selections: [CommunicationSelection] = []
    /// Clears tapped so far, in order.
    @Published private(set) var clears: [CommunicationClear] = []
    /// Indices into `selections` of the words currently on the strip, in
    /// order. A Clear selection pops the last one; the entry itself stays in
    /// `selections`, so the log keeps every pick the participant made.
    @Published private(set) var composedIndices: [Int] = []
    /// Index into `wordSet.target` of the word still needed. Equals
    /// `target.count` once the sentence is complete.
    @Published private(set) var expectedIndex: Int = 0
    /// Cell currently accumulating dwell, or nil.
    @Published private(set) var activeCellIndex: Int?
    /// `[0, 1]` dwell accumulated on `activeCellIndex`.
    @Published private(set) var dwellProgress: Double = 0
    /// Cell index of the most recent selection, for a brief visual flash.
    @Published private(set) var lastSelectedCell: Int?
    @Published private(set) var result: CommunicationTaskResult?

    let wordSet: CommunicationWordSet
    let screenSize: CGSize
    let calibration: CalibrationModel
    let dwellRequirement: CFTimeInterval
    /// Height of the composed-sentence strip along the top of the screen.
    /// The word grid is laid out *below* it so a fixation on the strip can
    /// never be scored as a word selection.
    let sentenceBarHeight: CGFloat
    /// Height reserved at the bottom for the Cancel / Finish controls, for
    /// the same reason.
    let controlBarHeight: CGFloat
    /// The rect the grid is laid out inside.
    let gridArea: CGRect

    private let audio: WordAudioPlayer
    private var runStart: CFTimeInterval = 0
    private var lastSelectionTime: CFTimeInterval?
    private var dwellCell: Int?
    private var dwellStart: CFTimeInterval = 0
    /// Cell that was just selected; blocked from re-selection until the gaze
    /// leaves it. Without this, holding a fixation emits the same word every
    /// `dwellRequirement` seconds.
    private var blockedCell: Int?
    /// True once the target sentence has been reproduced in order at least
    /// once. Tracked separately from `expectedIndex` because a clear rewinds
    /// that counter, and a sentence that was completed before the clear was
    /// still completed.
    private var sentenceEverCompleted = false

    private var runLog: ExperimentRunLog
    private(set) var runBundleURL: URL?

    /// Invoked once when the run completes, on the main actor. A run ends
    /// from the Finish button, off the gaze path, so the owner cannot rely on
    /// seeing `.complete` during its next `ingest`.
    var onComplete: ((CommunicationTaskResult) -> Void)?

    init(screenSize: CGSize,
         calibration: CalibrationModel,
         wordSet: CommunicationWordSet = .standard,
         audio: WordAudioPlayer,
         dwellRequirement: CFTimeInterval = 1.0,
         sentenceBarHeight: CGFloat = 96,
         controlBarHeight: CGFloat = 84) {
        self.screenSize = screenSize
        self.calibration = calibration
        self.wordSet = wordSet
        self.audio = audio
        self.dwellRequirement = dwellRequirement
        self.sentenceBarHeight = sentenceBarHeight
        self.controlBarHeight = controlBarHeight
        let area = CGRect(
            x: 0,
            y: sentenceBarHeight,
            width: screenSize.width,
            height: max(0, screenSize.height - sentenceBarHeight - controlBarHeight))
        self.gridArea = area
        self.runLog = ExperimentRunLog(
            experiment: "exp3",
            variant: "communication_task",
            runLabel: "Experiment 3 (communication task)",
            predictionSource: "smoothed",
            screenSize: screenSize,
            rows: wordSet.rows,
            cols: wordSet.cols,
            timing: ["dwell_requirement_s": dwellRequirement])

        // Shuffle word→cell assignment each run so the participant can't
        // learn positions across runs, which would turn a search task into a
        // memory task and inflate the rate. The Clear tile is exempt: it is
        // pinned to the bottom-left cell and the words fill the rest.
        let nRows = wordSet.rows
        let nCols = wordSet.cols
        let targetWords = Set(wordSet.target)
        let capacity = nRows * nCols
        let clearIdx = wordSet.clearCellIndex
        var words = wordSet.allWords.shuffled()
        // Pad or trim to exactly fill the cells the words get.
        let wordCapacity = capacity - 1
        if words.count > wordCapacity {
            words = Array(words.prefix(wordCapacity))
        }
        var wordQueue = words[...]
        let cellW = area.width / CGFloat(nCols)
        let cellH = area.height / CGFloat(nRows)
        self.cells = (0..<capacity).map { idx -> CommunicationCell in
            let r = idx / nCols
            let c = idx % nCols
            let word: String
            if idx == clearIdx {
                word = CommunicationWordSet.clearWord
            } else {
                word = wordQueue.popFirst() ?? ""
            }
            return CommunicationCell(
                index: idx,
                row: r,
                col: c,
                word: word,
                rect: CGRect(x: area.minX + CGFloat(c) * cellW,
                             y: area.minY + CGFloat(r) * cellH,
                             width: cellW,
                             height: cellH),
                isTargetWord: targetWords.contains(word))
        }
    }

    /// The word the sentence needs next, or nil once complete.
    var expectedWord: String? {
        expectedIndex < wordSet.target.count
            ? wordSet.target[expectedIndex] : nil
    }

    /// Every word selected so far, in order — the full record.
    var composedSentence: [String] { selections.map(\.word) }

    /// Words on the strip: the selections that have not been cleared.
    var visibleSelections: [CommunicationSelection] {
        composedIndices.compactMap { idx in
            idx < selections.count ? selections[idx] : nil
        }
    }

    func start() {
        guard !cells.isEmpty else {
            phase = .failed("No words")
            return
        }
        guard screenSize.width > 0, screenSize.height > 0 else {
            phase = .failed("No screen size")
            return
        }
        guard gridArea.height > 0, gridArea.width > 0 else {
            phase = .failed("Screen too small for the word grid")
            return
        }
        // Every target word must actually be on the grid, or the run is
        // unwinnable. This can only happen if the word set is misconfigured
        // (more words than cells), so fail loudly rather than collecting a
        // run that can never complete.
        let onGrid = Set(cells.map(\.word))
        let missing = wordSet.target.filter { !onGrid.contains($0) }
        guard missing.isEmpty else {
            phase = .failed("Target words not on grid: \(missing.joined(separator: ", "))")
            return
        }
        selections.removeAll()
        clears.removeAll()
        composedIndices = []
        expectedIndex = 0
        activeCellIndex = nil
        dwellCell = nil
        blockedCell = nil
        lastSelectedCell = nil
        sentenceEverCompleted = false
        dwellProgress = 0
        runStart = CACurrentMediaTime()
        lastSelectionTime = nil
        phase = .running
        runLog.begin()
    }

    func cancel() {
        audio.stop()
        phase = .idle
        activeCellIndex = nil
        dwellProgress = 0
        result = nil
    }

    /// Take the last word back off the composed-sentence strip — a backspace,
    /// not a reset: the rest of the sentence stands.
    ///
    /// This is a *display* retraction, not an undo of the data: the cleared
    /// selection stays in `selections` and keeps counting towards the error
    /// rate, because it happened. What it gives the participant is a way to
    /// fix one mis-pick without losing the words already composed.
    /// Triggered by dwelling on the Clear grid tile.
    private func performClear() {
        guard let removedIdx = composedIndices.popLast(),
              removedIdx < selections.count else {
            return
        }
        let removed = selections[removedIdx]
        clears.append(CommunicationClear(
            afterSelection: selections.count,
            removedWord: removed.word,
            removedWasCorrect: removed.isCorrect,
            atSeconds: CACurrentMediaTime() - runStart))
        // Only a correct word had advanced the sentence, so only a correct
        // word rewinds it — clearing a mis-pick leaves the participant on the
        // same target word they were already trying to say.
        if removed.isCorrect {
            expectedIndex = max(0, expectedIndex - 1)
        }
        lastSelectedCell = composedIndices.last.map { selections[$0].cellIndex }
        // The next dwell starts fresh — otherwise a gaze already parked on a
        // cell when Clear fired could immediately select again.
        blockedCell = nil
        dwellStart = CACurrentMediaTime()
        resetDwell()
        audio.speak("cleared \(removed.word)")
    }

    /// End the run, scoring whatever was composed. The only way a run
    /// finishes — there is no timer and no selection cap behind it.
    func finishNow() {
        guard phase == .running else { return }
        finish()
    }

    func ingest(predictionScreenAbs: CGPoint,
                gazeCam: simd_double3,
                headPose: HeadPose,
                pupilDiameters: PupilMeasure.Diameters = .init(leftPx: .nan, rightPx: .nan),
                diagnostics: ExperimentRunLog.Diagnostics = .empty) {
        guard phase == .running else { return }
        let now = CACurrentMediaTime()

        let hitCell = cells.first { $0.rect.contains(predictionScreenAbs) }

        // The optional-coalescing expressions are hoisted into typed locals:
        // inlining them into this 18-argument initializer pushes Swift's
        // expression type-checker into an exponential blowup.
        let e = headPose.euler
        let logIdx: Int = hitCell?.index ?? -1
        let logRow: Int = hitCell?.row ?? -1
        let logCol: Int = hitCell?.col ?? -1
        let logRect: CGRect = hitCell?.rect ?? .null
        let order: Int = selections.count + 1
        runLog.append(.init(
            t: now,
            trialOrder: order,
            cellIdx: logIdx,
            row: logRow,
            col: logCol,
            phase: "active",
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

        guard let cell = hitCell else {
            // Gaze is off-grid (only possible in gaps/rounding) — drop any
            // open dwell and clear the re-selection block.
            resetDwell()
            blockedCell = nil
            return
        }

        // Leaving the just-selected cell re-arms it.
        if let blocked = blockedCell, blocked != cell.index {
            blockedCell = nil
        }
        if blockedCell == cell.index {
            // Still sitting on the word we just spoke: no dwell accrues.
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

        let held = now - dwellStart
        dwellProgress = max(0, min(1, held / dwellRequirement))
        if held >= dwellRequirement {
            select(cell, at: now, prediction: predictionScreenAbs)
        }
    }

    private func resetDwell() {
        if dwellCell != nil {
            dwellCell = nil
            activeCellIndex = nil
            dwellProgress = 0
        }
    }

    private func select(_ cell: CommunicationCell,
                        at now: CFTimeInterval,
                        prediction: CGPoint) {
        if cell.isClear {
            // Not a word: it takes the last word back off the strip and is
            // never scored. Block it the same way a word is, so a gaze parked
            // on it deletes one word rather than the whole sentence.
            performClear()
            blockedCell = cell.index
            resetDwell()
            return
        }
        let isCorrect = cell.word == expectedWord
        let selection = CommunicationSelection(
            order: selections.count + 1,
            cellIndex: cell.index,
            word: cell.word,
            isCorrect: isCorrect,
            targetPosition: isCorrect ? expectedIndex : -1,
            atSeconds: now - runStart,
            sincePreviousSeconds: now - (lastSelectionTime ?? runStart),
            predX: Double(prediction.x),
            predY: Double(prediction.y))
        selections.append(selection)
        composedIndices.append(selections.count - 1)
        lastSelectionTime = now
        lastSelectedCell = cell.index

        // Audio feedback for every selection, right or wrong — the
        // participant needs to hear what they actually picked in order to
        // correct it, which is the whole point of the feedback channel.
        audio.speak(cell.word)

        if isCorrect {
            expectedIndex += 1
            if expectedIndex >= wordSet.target.count { sentenceEverCompleted = true }
        }

        blockedCell = cell.index
        resetDwell()

        // No automatic stop. The run ends only when the operator says so —
        // Finish (score and log it) or Cancel (discard it). Reaching the end
        // of the target sentence is a *milestone*, not a terminator: the
        // participant keeps composing, and every further selection is
        // recorded with `targetPosition = -1`.
    }

    private func finish() {
        let completed = sentenceEverCompleted
        var r = CommunicationTaskResult(
            wordSet: wordSet,
            cells: cells,
            selections: selections,
            clears: clears,
            screenSize: screenSize,
            distancePoints: VisualAngle.distancePoints(for: calibration),
            durationSeconds: CACurrentMediaTime() - runStart,
            completed: completed)
        do {
            let out = try runLog.finish(
                trialsCSV: r.csvString(),
                summary: [
                    "completed": completed,
                    "n_selections": selections.count,
                    "n_clears": clears.count,
                    "correct": r.correctCount,
                    "incorrect": r.incorrectCount,
                    "selection_accuracy_pct": r.selectionAccuracy * 100.0,
                    "mean_s_per_selection": r.meanSecondsPerSelection,
                    "mean_s_per_correct_word": r.meanSecondsPerCorrectWord,
                    "words_per_min": r.wordsPerMinute,
                ])
            self.runBundleURL = out.zip
            r.runBundleURL = out.zip
            // Join this launch's session, so the operator can ship every run
            // of this experiment as one archive without hand-filtering.
            ExperimentSession.shared.record(experiment: "exp3",
                                            label: "communication",
                                            directory: out.directory)
        } catch {
            print("experiment3 communication run bundle failed: \(error)")
        }
        self.result = r
        self.activeCellIndex = nil
        phase = .complete
        onComplete?(r)
    }
}
