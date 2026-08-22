import Foundation
import simd
import QuartzCore
import CoreGraphics
import UIKit

/// Experiment 1 — grid-focus accuracy. The screen is divided into an
/// `rows × cols` grid; one cell at a time is highlighted in red and the user
/// is asked to focus on it for a fixed dwell + capture window. After every
/// cell has been visited (in shuffled order, with optional repeats) the
/// system reports a hit-rate: the fraction of trials whose mean predicted
/// gaze point fell inside the highlighted cell.

/// A single grid layout used for one experiment run. `repeats > 1` cycles
/// through the shuffled cell list multiple times (used by the 3×3 case
/// which doubles up to fill its 1-minute budget).
struct GridSize: Identifiable, Hashable {
    let rows: Int
    let cols: Int
    let repeats: Int
    let label: String
    let estimatedMinutes: Double
    let walking: Bool

    var id: String { "\(rows)x\(cols)-r\(repeats)\(walking ? "-w" : "")" }
    var cellCount: Int { rows * cols }
    var trialCount: Int { cellCount * repeats }

    /// Grid resolutions Experiment 1 can sweep, coarse → fine.
    ///
    /// 3×3 repeats twice so its 18 trials are comparable in count to the
    /// finer grids; with only 9 trials its per-cell accuracy would carry
    /// visibly wider error bars than the rest.
    static let presets: [GridSize] = [
        .init(rows: 3, cols: 3, repeats: 2, label: "3×3 (×2)",  estimatedMinutes: 1.0, walking: false),
        .init(rows: 4, cols: 4, repeats: 1, label: "4×4",       estimatedMinutes: 1.0, walking: false),
        .init(rows: 5, cols: 4, repeats: 1, label: "5×4",       estimatedMinutes: 1.0, walking: false),
        .init(rows: 6, cols: 4, repeats: 1, label: "6×4",       estimatedMinutes: 1.5, walking: false),
        .init(rows: 6, cols: 9, repeats: 1, label: "6×9",       estimatedMinutes: 3.0, walking: false),
        .init(rows: 9, cols: 9, repeats: 1, label: "9×9",       estimatedMinutes: 5.0, walking: false),
        .init(rows: 9, cols: 9, repeats: 1, label: "9×9 walking", estimatedMinutes: 5.0, walking: true),
    ]
}

/// A per-frame sample captured during one trial's 2 s capture window.
struct GridFrameSample {
    let t: CFTimeInterval
    let gazeCam: simd_double3
    let headYawDeg: Double
    let headPitchDeg: Double
    let headRollDeg: Double
    let headTx: Double
    let headTy: Double
    let headTz: Double
    /// Calibrated prediction in screen-**center-relative** points.
    let prediction: simd_double2
}

/// Outcome of one trial (one highlighted cell). Geometry is stored in
/// absolute screen points so the results view doesn't need the screen size
/// to recompute hit/miss.
struct GridTrial {
    /// Trial number, 1-based, in presentation order.
    let order: Int
    /// Cell index in row-major order, 0 ≤ idx < rows*cols.
    let cellIdx: Int
    let row: Int
    let col: Int
    /// Cell rect in absolute screen coords (top-left origin).
    let rect: CGRect
    /// Cell center in absolute screen coords.
    let center: CGPoint
    /// Mean predicted point in absolute screen coords (= centerOfScreen +
    /// mean of center-relative samples).
    let meanPredictionAbs: CGPoint
    /// Did `meanPredictionAbs` fall inside `rect`?
    let isHit: Bool
    /// Euclidean distance from `meanPredictionAbs` to the cell center, in points.
    let errorPoints: Double
    /// Number of frames captured during the trial.
    let sampleCount: Int
    /// Last per-frame sample of the trial (head pose / raw gaze at capture
    /// end). Convenient for the CSV without exploding the row count.
    let lastSample: GridFrameSample?
}

/// Accuracy for one grid cell across every trial in which it was the target.
/// With `repeats > 1` a cell is visited more than once, so "per-cell accuracy"
/// is a rate rather than a single hit/miss.
struct GridCellAccuracy: Identifiable {
    let cellIdx: Int
    let row: Int
    let col: Int
    let attempts: Int
    let hits: Int
    /// Mean of this cell's trial errors, screen points.
    let meanErrorPoints: Double

    var id: Int { cellIdx }
    var accuracy: Double {
        attempts == 0 ? .nan : Double(hits) / Double(attempts)
    }
}

struct GridExperimentResult {
    let gridSize: GridSize
    let screenSize: CGSize
    let trials: [GridTrial]
    /// Virtual eye-to-screen distance in points, from the calibration the run
    /// was scored against. Drives every degrees figure below.
    var distancePoints: Double = .nan
    /// Set only when the run was a fine-tune collection (i.e. a
    /// `FineTuneDataCollector` was attached to the controller). Points to
    /// the zipped per-frame dump in `Documents/finetune_runs/`.
    var fineTuneBundleURL: URL? = nil
    /// Zipped `Documents/experiment_runs/` bundle for this run (samples.csv
    /// + trials.csv + meta.json). Nil only if writing or zipping failed.
    var runBundleURL: URL? = nil

    var hitCount: Int { trials.filter(\.isHit).count }
    var accuracy: Double {
        trials.isEmpty ? .nan : Double(hitCount) / Double(trials.count)
    }
    var meanErrorPoints: Double {
        let errs = trials.map(\.errorPoints).filter { $0.isFinite }
        guard !errs.isEmpty else { return .nan }
        return errs.reduce(0, +) / Double(errs.count)
    }

    /// Mean error in degrees of visual angle. This is the figure that can be
    /// compared across grid resolutions — px error is resolution-independent
    /// too, but degrees is what the eye-tracking literature quotes.
    var meanErrorDegrees: Double {
        VisualAngle.meanDegrees(pointErrors: trials.map(\.errorPoints),
                                distancePoints: distancePoints)
    }

    /// Cell size in points, and the angle it subtends. A grid resolution is
    /// only usable if the mean error is comfortably smaller than this — which
    /// is the whole "accuracy vs. resolution" trade-off, in one number.
    var cellSize: CGSize {
        CGSize(width: screenSize.width / CGFloat(gridSize.cols),
               height: screenSize.height / CGFloat(gridSize.rows))
    }

    /// Angle subtended by half the smaller cell dimension — the error budget
    /// a prediction has before it lands in a neighbouring cell.
    var cellToleranceDegrees: Double {
        let half = Double(min(cellSize.width, cellSize.height)) * 0.5
        return VisualAngle.degrees(points: half, distancePoints: distancePoints)
    }

    /// Per-cell accuracy, ordered row-major over the cells that were tested.
    var perCellAccuracy: [GridCellAccuracy] {
        var attempts: [Int: Int] = [:]
        var hits: [Int: Int] = [:]
        var errSum: [Int: Double] = [:]
        var errN: [Int: Int] = [:]
        for t in trials {
            attempts[t.cellIdx, default: 0] += 1
            if t.isHit { hits[t.cellIdx, default: 0] += 1 }
            if t.errorPoints.isFinite {
                errSum[t.cellIdx, default: 0] += t.errorPoints
                errN[t.cellIdx, default: 0] += 1
            }
        }
        return attempts.keys.sorted().map { idx in
            let n = errN[idx] ?? 0
            return GridCellAccuracy(
                cellIdx: idx,
                row: idx / gridSize.cols,
                col: idx % gridSize.cols,
                attempts: attempts[idx] ?? 0,
                hits: hits[idx] ?? 0,
                meanErrorPoints: n > 0 ? (errSum[idx] ?? 0) / Double(n) : .nan)
        }
    }

    /// Master log file (shared across all experiment runs).
    static let masterLogFilename = "grid_experiment_log.csv"
    static func masterLogURL() -> URL {
        FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(masterLogFilename)
    }

    /// Build the CSV body for this run (no run-header line — that's added
    /// by `appendToMasterLog`).
    func csvString() -> String {
        var lines: [String] = []
        lines.append([
            "trial", "cell_idx", "row", "col",
            "cell_x_min", "cell_y_min", "cell_x_max", "cell_y_max",
            "cell_cx", "cell_cy",
            "pred_x", "pred_y", "err_pt", "err_deg", "hit",
            "gaze_cam_x", "gaze_cam_y", "gaze_cam_z",
            "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
            "head_tx_mm", "head_ty_mm", "head_tz_mm",
            "n_samples"
        ].joined(separator: ","))

        for t in trials {
            let g = t.lastSample
            let row: [String] = [
                "\(t.order)", "\(t.cellIdx)", "\(t.row)", "\(t.col)",
                fmt(Double(t.rect.minX), 2), fmt(Double(t.rect.minY), 2),
                fmt(Double(t.rect.maxX), 2), fmt(Double(t.rect.maxY), 2),
                fmt(Double(t.center.x), 2), fmt(Double(t.center.y), 2),
                fmt(Double(t.meanPredictionAbs.x), 2),
                fmt(Double(t.meanPredictionAbs.y), 2),
                fmt(t.errorPoints, 2),
                fmt(VisualAngle.degrees(points: t.errorPoints,
                                        distancePoints: distancePoints), 3),
                t.isHit ? "1" : "0",
                g.map { fmt($0.gazeCam.x, 6) } ?? "",
                g.map { fmt($0.gazeCam.y, 6) } ?? "",
                g.map { fmt($0.gazeCam.z, 6) } ?? "",
                g.map { fmt($0.headYawDeg, 3) } ?? "",
                g.map { fmt($0.headPitchDeg, 3) } ?? "",
                g.map { fmt($0.headRollDeg, 3) } ?? "",
                g.map { fmt($0.headTx, 2) } ?? "",
                g.map { fmt($0.headTy, 2) } ?? "",
                g.map { fmt($0.headTz, 2) } ?? "",
                "\(t.sampleCount)"
            ]
            lines.append(row.joined(separator: ","))
        }

        lines.append("")
        lines.append("# Per-cell accuracy")
        lines.append("cell_idx,row,col,attempts,hits,accuracy_pct,mean_err_pt,mean_err_deg")
        for c in perCellAccuracy {
            var row: [String] = []
            row.append("\(c.cellIdx)")
            row.append("\(c.row)")
            row.append("\(c.col)")
            row.append("\(c.attempts)")
            row.append("\(c.hits)")
            row.append(fmt(c.accuracy * 100.0, 2))
            row.append(fmt(c.meanErrorPoints, 2))
            row.append(fmt(VisualAngle.degrees(points: c.meanErrorPoints,
                                               distancePoints: distancePoints), 3))
            lines.append(row.joined(separator: ","))
        }

        lines.append("")
        lines.append("# Overall")
        lines.append("rows,cols,repeats,trial_count,hits,accuracy_pct,mean_error_pt,mean_error_deg,cell_w_pt,cell_h_pt,cell_tolerance_deg,tz_points,screen_w_pt,screen_h_pt,walking")
        var o: [String] = []
        o.append("\(gridSize.rows)")
        o.append("\(gridSize.cols)")
        o.append("\(gridSize.repeats)")
        o.append("\(trials.count)")
        o.append("\(hitCount)")
        o.append(fmt(accuracy * 100.0, 2))
        o.append(fmt(meanErrorPoints, 2))
        o.append(fmt(meanErrorDegrees, 3))
        o.append(fmt(Double(cellSize.width), 2))
        o.append(fmt(Double(cellSize.height), 2))
        o.append(fmt(cellToleranceDegrees, 3))
        o.append(fmt(distancePoints, 2))
        o.append(fmt(Double(screenSize.width), 2))
        o.append(fmt(Double(screenSize.height), 2))
        o.append(gridSize.walking ? "1" : "0")
        lines.append(o.joined(separator: ","))

        return lines.joined(separator: "\n") + "\n"
    }

    private func fmt(_ x: Double, _ places: Int) -> String {
        guard x.isFinite else { return "" }
        return String(format: "%.\(places)f", x)
    }

    /// Append this run to `grid_experiment_log.csv`. Numbered by a
    /// `grid_next_run_id` counter in UserDefaults.
    @discardableResult
    func appendToMasterLog() throws -> URL {
        let url = Self.masterLogURL()
        let runId = UserDefaults.standard.integer(forKey: "grid_next_run_id") + 1
        UserDefaults.standard.set(runId, forKey: "grid_next_run_id")

        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let stamp = df.string(from: Date())

        var section = "\n=== Run \(runId) @ \(stamp) — \(gridSize.label) ===\n"
        section += csvString()

        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            if let data = section.data(using: .utf8) {
                try handle.write(contentsOf: data)
            }
        } else {
            let first = section.hasPrefix("\n") ? String(section.dropFirst()) : section
            try first.write(to: url, atomically: true, encoding: .utf8)
        }
        return url
    }
}

/// Drives one grid-experiment run: shuffles cell order, walks through
/// dwell → capture per trial, finalizes a `GridExperimentResult`.
@MainActor
final class GridExperimentController: ObservableObject {

    enum Phase: Equatable {
        case idle
        case dwelling
        case capturing
        case complete
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var trialIndex: Int = 0
    @Published private(set) var trialProgress: Double = 0
    @Published private(set) var result: GridExperimentResult?

    let gridSize: GridSize
    let screenSize: CGSize
    let calibration: CalibrationModel
    let preDuration: CFTimeInterval
    let captureDuration: CFTimeInterval

    /// Optional sink for the off-device fine-tune flow. When non-nil, each
    /// capture-window frame's face image + metadata is dumped to disk.
    var dataCollector: FineTuneDataCollector?
    /// Zip URL produced by `dataCollector?.finalize()` on completion.
    /// Surfaced to the results view so the user can share it.
    private(set) var fineTuneBundleURL: URL?
    /// Zip of this run's `experiment_runs/` bundle (samples + trials + meta).
    private(set) var runBundleURL: URL?

    /// Shuffled cell indices in presentation order (length = trialCount).
    let cellOrder: [Int]

    /// Whole-run per-frame log (dwell + capture, every trial), written as
    /// an `experiment_runs/` bundle on completion.
    private var runLog: ExperimentRunLog

    private var perTrialSamples: [[GridFrameSample]]
    private var trialStart: CFTimeInterval = 0
    private var timer: Timer?

    init(gridSize: GridSize,
         screenSize: CGSize,
         calibration: CalibrationModel,
         preDuration: CFTimeInterval = 1.0,
         captureDuration: CFTimeInterval = 2.0) {
        self.gridSize = gridSize
        self.screenSize = screenSize
        self.calibration = calibration
        self.preDuration = preDuration
        self.captureDuration = captureDuration

        // One shuffled pass per repeat. We re-shuffle each pass so the user
        // doesn't see the same order twice in the 3×3 doubled case.
        var order: [Int] = []
        for _ in 0..<max(1, gridSize.repeats) {
            order.append(contentsOf: (0..<gridSize.cellCount).shuffled())
        }
        self.cellOrder = order
        self.perTrialSamples = Array(repeating: [], count: order.count)

        // Variant key must be filename-safe: the label uses "×" and spaces.
        let variant = "\(gridSize.rows)x\(gridSize.cols)"
            + (gridSize.repeats > 1 ? "r\(gridSize.repeats)" : "")
            + (gridSize.walking ? "_walking" : "")
        self.runLog = ExperimentRunLog(
            experiment: "exp1",
            variant: variant,
            runLabel: gridSize.label,
            // Exp 1's trial mean is built from the raw calibrated point, so
            // that's what the samples must hold for the mean to be
            // reproducible off-device.
            predictionSource: "calibrated_raw",
            screenSize: screenSize,
            rows: gridSize.rows,
            cols: gridSize.cols,
            timing: ["dwell_s": preDuration, "capture_s": captureDuration])
    }

    deinit { timer?.invalidate() }

    func start() {
        guard !cellOrder.isEmpty else {
            phase = .failed("Empty grid")
            return
        }
        guard screenSize.width > 0, screenSize.height > 0 else {
            phase = .failed("No screen size")
            return
        }
        timer?.invalidate()
        trialIndex = 0
        trialStart = CACurrentMediaTime()
        trialProgress = 0
        phase = .dwelling
        runLog.begin()
        let t = Timer(timeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
        RunLoop.main.add(t, forMode: .common)
        self.timer = t
    }

    func cancel() {
        timer?.invalidate()
        timer = nil
        phase = .idle
        trialProgress = 0
        result = nil
    }

    /// Feed a raw gaze sample + head pose. Stored only during the capture
    /// window of the current trial.
    ///
    /// The trailing optional arguments carry the per-frame state the
    /// off-device fine-tune flow needs (the face image the CNN saw, the
    /// virtual-camera rotation `R_n`, and the eye position in camera
    /// coords from PnP). They're nil for plain Exp 1 runs and only used
    /// when `dataCollector != nil`.
    func ingest(gazeCam: simd_double3,
                headPose: HeadPose,
                faceImage: UIImage? = nil,
                normalizationRotation: simd_double3x3? = nil,
                eyePositionCam: simd_double3? = nil,
                pupilDiameters: PupilMeasure.Diameters = .init(leftPx: .nan, rightPx: .nan),
                diagnostics: ExperimentRunLog.Diagnostics = .empty) {
        guard phase == .capturing || phase == .dwelling else { return }
        guard trialIndex < perTrialSamples.count else { return }
        let p = calibration.predict(gazeCam: gazeCam)
        guard p.x.isFinite, p.y.isFinite else { return }
        let e = headPose.euler

        // Whole-run frame log: every frame of every trial, dwell included.
        let logCellIdx = cellOrder[trialIndex]
        let logRect = Self.rect(forCellIndex: logCellIdx,
                                rows: gridSize.rows, cols: gridSize.cols,
                                screenSize: screenSize)
        runLog.append(.init(
            t: CACurrentMediaTime(),
            trialOrder: trialIndex + 1,
            cellIdx: logCellIdx,
            row: logCellIdx / gridSize.cols,
            col: logCellIdx % gridSize.cols,
            phase: phase == .capturing ? "capture" : "dwell",
            cellRect: logRect,
            predX: Double(screenSize.width) * 0.5 + p.x,
            predY: Double(screenSize.height) * 0.5 + p.y,
            headYawDeg: e.yaw,
            headPitchDeg: e.pitch,
            headRollDeg: e.roll,
            headTx: headPose.translation.x,
            headTy: headPose.translation.y,
            headTz: headPose.translation.z,
            pupilLeftPx: pupilDiameters.leftPx,
            pupilRightPx: pupilDiameters.rightPx,
            diag: diagnostics))

        // Hit computation + fine-tune collection stay capture-only: dwell
        // frames are mid-saccade and would poison both the trial mean and
        // the fine-tune labels.
        guard phase == .capturing else { return }

        perTrialSamples[trialIndex].append(GridFrameSample(
            t: CACurrentMediaTime(),
            gazeCam: gazeCam,
            headYawDeg: e.yaw,
            headPitchDeg: e.pitch,
            headRollDeg: e.roll,
            headTx: headPose.translation.x,
            headTy: headPose.translation.y,
            headTz: headPose.translation.z,
            prediction: p
        ))

        if let collector = dataCollector,
           let img = faceImage,
           let Rn = normalizationRotation,
           let eye = eyePositionCam,
           trialIndex < cellOrder.count {
            let cellIdx = cellOrder[trialIndex]
            let rect = Self.rect(forCellIndex: cellIdx,
                                 rows: gridSize.rows, cols: gridSize.cols,
                                 screenSize: screenSize)
            let center = CGPoint(x: rect.midX, y: rect.midY)
            collector.write(trialIdx: trialIndex,
                            cellIdx: cellIdx,
                            cellCenter: center,
                            faceImage: img,
                            normalizationRotation: Rn,
                            eyePositionCam: eye,
                            headPose: headPose,
                            gazePredCam: gazeCam,
                            pupilDiameters: pupilDiameters)
        }
    }

    /// Current cell rect in absolute screen coords (top-left origin), or
    /// `nil` if there is no current trial.
    var currentCellRect: CGRect? {
        guard trialIndex < cellOrder.count else { return nil }
        return Self.rect(forCellIndex: cellOrder[trialIndex],
                         rows: gridSize.rows, cols: gridSize.cols,
                         screenSize: screenSize)
    }

    static func rect(forCellIndex idx: Int,
                     rows: Int, cols: Int,
                     screenSize: CGSize) -> CGRect {
        let row = idx / cols
        let col = idx % cols
        let cellW = screenSize.width  / CGFloat(cols)
        let cellH = screenSize.height / CGFloat(rows)
        return CGRect(x: CGFloat(col) * cellW,
                      y: CGFloat(row) * cellH,
                      width:  cellW,
                      height: cellH)
    }

    private func tick() {
        guard phase == .dwelling || phase == .capturing else { return }
        let elapsed = CACurrentMediaTime() - trialStart
        let total = preDuration + captureDuration
        trialProgress = max(0, min(1, elapsed / total))

        switch phase {
        case .dwelling where elapsed >= preDuration:
            phase = .capturing
        case .capturing where elapsed >= total:
            advance()
        default:
            break
        }
    }

    private func advance() {
        let next = trialIndex + 1
        if next < cellOrder.count {
            trialIndex = next
            trialStart = CACurrentMediaTime()
            trialProgress = 0
            phase = .dwelling
        } else {
            timer?.invalidate()
            timer = nil
            finish()
        }
    }

    private func finish() {
        var trials: [GridTrial] = []
        trials.reserveCapacity(cellOrder.count)
        let centerX = Double(screenSize.width) * 0.5
        let centerY = Double(screenSize.height) * 0.5

        for (i, idx) in cellOrder.enumerated() {
            let rect = Self.rect(forCellIndex: idx,
                                 rows: gridSize.rows, cols: gridSize.cols,
                                 screenSize: screenSize)
            let center = CGPoint(x: rect.midX, y: rect.midY)
            let samples = perTrialSamples[i]
            let predsAbs: CGPoint
            if samples.isEmpty {
                predsAbs = CGPoint(x: CGFloat.nan, y: CGFloat.nan)
            } else {
                let n = Double(samples.count)
                let sx = samples.reduce(0.0) { $0 + $1.prediction.x } / n
                let sy = samples.reduce(0.0) { $0 + $1.prediction.y } / n
                predsAbs = CGPoint(x: centerX + sx, y: centerY + sy)
            }
            let hit = !samples.isEmpty && rect.contains(predsAbs)
            let err: Double
            if samples.isEmpty {
                err = .nan
            } else {
                let dx = Double(predsAbs.x - center.x)
                let dy = Double(predsAbs.y - center.y)
                err = sqrt(dx * dx + dy * dy)
            }
            trials.append(GridTrial(
                order: i + 1,
                cellIdx: idx,
                row: idx / gridSize.cols,
                col: idx % gridSize.cols,
                rect: rect,
                center: center,
                meanPredictionAbs: predsAbs,
                isHit: hit,
                errorPoints: err,
                sampleCount: samples.count,
                lastSample: samples.last
            ))
        }

        // Close + zip the fine-tune bundle, if collection was active. The
        // URL also gets stashed on the controller so a hung result-view
        // re-renders still find the bundle.
        let bundle = dataCollector?.finalize()
        self.fineTuneBundleURL = bundle

        var r = GridExperimentResult(
            gridSize: gridSize,
            screenSize: screenSize,
            trials: trials,
            distancePoints: VisualAngle.distancePoints(for: calibration)
        )
        r.fineTuneBundleURL = bundle

        // Per-run bundle. Written after the trials are assembled so
        // trials.csv and meta.json's summary come from the same result.
        do {
            let out = try runLog.finish(
                trialsCSV: r.csvString(),
                summary: [
                    "trial_count": trials.count,
                    "hits": r.hitCount,
                    "accuracy_pct": r.accuracy * 100.0,
                    "mean_error_pt": r.meanErrorPoints,
                    "mean_error_deg": r.meanErrorDegrees,
                    "cell_tolerance_deg": r.cellToleranceDegrees,
                    "repeats": gridSize.repeats,
                    "walking": gridSize.walking,
                ])
            self.runBundleURL = out.zip
            r.runBundleURL = out.zip
        } catch {
            print("grid experiment run bundle failed: \(error)")
        }

        self.result = r
        phase = .complete
    }
}
