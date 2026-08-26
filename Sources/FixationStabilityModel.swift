import Foundation
import simd
import QuartzCore
import CoreGraphics

/// Experiment 2 — fixation stability across a 4×4 grid.
///
/// Stimulus: a 4×4 grid of boxes. One cell at a time holds a single red circle
/// at its centre; the participant fixates that circle for a fixed duration,
/// then the circle moves to the next cell (shuffled order) and the same
/// measurement repeats. Every cell is visited once.
///
/// This isolates *precision* from *accuracy*, and does so **as a function of
/// screen position**. Experiment 1 moves a target around and so conflates the
/// two: a miss there could be systematic bias or could be jitter. Holding one
/// point separates them — the centroid's offset from the circle is the bias,
/// and the spread about that centroid is the noise floor of the whole pipeline
/// (CNN + PnP, measured *before* any output smoothing — see `ingest`). Repeating that at 16 positions shows whether the
/// noise floor is uniform or degrades toward the corners, which a single
/// centred box cannot reveal. No grid experiment can resolve better than the
/// noise floor reported here, which is what makes this the measurement that
/// bounds all the others.
///
/// Per-cell protocol:
///  1. `settleDuration` seconds of unscored lead-in, so the saccade onto the
///     circle stays out of the statistics.
///  2. `captureDuration` seconds of scored fixation.
///  3. Advance to the next cell; complete after the last one.

/// One scored frame of a cell's capture window.
struct FixationSample {
    /// Seconds since that cell's capture window opened.
    let t: Double
    /// Index of the cell this frame was scored against.
    let cellIdx: Int
    /// Prediction in screen-**center-relative** points.
    let prediction: simd_double2
    let headYawDeg: Double
    let headPitchDeg: Double
    let headRollDeg: Double
}

/// Per-cell outcome: one fixation held at one screen position.
struct FixationCell: Identifiable {
    /// Row-major cell index, 0 ..< rows*cols.
    let index: Int
    let row: Int
    let col: Int
    /// Cell box in absolute screen points.
    let rect: CGRect
    /// Circle centre in screen-**center-relative** points — the fixation
    /// target this cell's statistics are measured against.
    let target: simd_double2
    /// 1-based presentation order (cells are shuffled).
    var order: Int = 0
    /// Scored samples for this cell.
    var samples: [FixationSample] = []

    var id: Int { index }

    /// Circle centre in absolute screen points.
    var center: CGPoint { CGPoint(x: rect.midX, y: rect.midY) }

    var scatter: GazeScatter? {
        GazeScatter.compute(samples: samples.map(\.prediction), target: target)
    }

    var meanDeviationPoints: Double { scatter?.meanDeviation ?? .nan }
    /// RMS scatter about this cell's own centroid — the precision figure.
    var rmsPoints: Double { scatter?.rms ?? .nan }
    var standardDeviationPoints: Double { scatter?.standardDeviation ?? .nan }
    /// Distance from the centroid to the circle centre: the systematic
    /// component at this screen position, with jitter averaged out.
    var biasPoints: Double { scatter?.bias ?? .nan }
    var sampleCount: Int { samples.count }

    /// Fraction of this cell's frames that landed inside its own box.
    func containment(screenSize: CGSize) -> Double {
        guard !samples.isEmpty else { return .nan }
        let cx = Double(screenSize.width) * 0.5
        let cy = Double(screenSize.height) * 0.5
        let inside = samples.filter {
            rect.contains(CGPoint(x: cx + $0.prediction.x,
                                  y: cy + $0.prediction.y))
        }
        return Double(inside.count) / Double(samples.count)
    }
}

struct FixationStabilityResult {
    let rows: Int
    let cols: Int
    let screenSize: CGSize
    /// Cells in row-major order (not presentation order).
    let cells: [FixationCell]
    /// Scored seconds per cell.
    let captureDuration: Double
    /// Virtual eye-to-screen distance in points, for degree conversions.
    let distancePoints: Double
    var runBundleURL: URL? = nil

    /// Cells that collected at least one scored frame.
    var scoredCells: [FixationCell] { cells.filter { !$0.samples.isEmpty } }
    var scoredCellCount: Int { scoredCells.count }
    var totalSampleCount: Int { cells.reduce(0) { $0 + $1.samples.count } }

    // MARK: aggregate accuracy / precision
    //
    // Aggregates average the PER-CELL statistics rather than pooling every
    // sample into one cloud. Pooling would measure the spread of 16 different
    // fixation positions about a common centroid — a number dominated by the
    // grid geometry, not by tracker noise.

    /// Mean deviation from the circle centre, averaged over cells.
    var meanDeviationPoints: Double { meanOverCells(\.meanDeviationPoints) }
    var meanDeviationDegrees: Double {
        VisualAngle.meanDegrees(pointErrors: scoredCells.map(\.meanDeviationPoints),
                                distancePoints: distancePoints)
    }

    /// Mean RMS scatter (precision), averaged over cells.
    var rmsPoints: Double { meanOverCells(\.rmsPoints) }
    var rmsDegrees: Double {
        VisualAngle.meanDegrees(pointErrors: scoredCells.map(\.rmsPoints),
                                distancePoints: distancePoints)
    }

    /// Mean radial SD, averaged over cells.
    var standardDeviationPoints: Double {
        meanOverCells(\.standardDeviationPoints)
    }
    var standardDeviationDegrees: Double {
        VisualAngle.meanDegrees(
            pointErrors: scoredCells.map(\.standardDeviationPoints),
            distancePoints: distancePoints)
    }

    /// Mean bias, averaged over cells.
    var biasPoints: Double { meanOverCells(\.biasPoints) }
    var biasDegrees: Double {
        VisualAngle.meanDegrees(pointErrors: scoredCells.map(\.biasPoints),
                                distancePoints: distancePoints)
    }

    /// Worst single cell's precision. A tracker can look fine on average and
    /// still be unusable in one corner — exactly what a 4×4 sweep exposes and
    /// a single centred box cannot.
    var worstRmsDegrees: Double {
        let vals = scoredCells
            .map { VisualAngle.degrees(points: $0.rmsPoints,
                                       distancePoints: distancePoints) }
            .filter { $0.isFinite }
        return vals.max() ?? .nan
    }

    /// The cell holding that worst precision, for the results screen.
    var worstCell: FixationCell? {
        scoredCells
            .filter { $0.rmsPoints.isFinite }
            .max { $0.rmsPoints < $1.rmsPoints }
    }

    /// Per-axis SD, pooled within each cell then averaged across cells.
    /// Reported separately because vertical gaze error on a front-camera
    /// estimator is usually much worse than horizontal, and a single radial
    /// number hides that.
    var sdX: Double { meanOverCells { axisSD($0, \.x) } }
    var sdY: Double { meanOverCells { axisSD($0, \.y) } }

    /// Fraction of all scored frames that landed inside their own cell box.
    var containment: Double {
        let scored = totalSampleCount
        guard scored > 0 else { return .nan }
        let cx = Double(screenSize.width) * 0.5
        let cy = Double(screenSize.height) * 0.5
        var inside = 0
        for c in cells {
            for s in c.samples
            where c.rect.contains(CGPoint(x: cx + s.prediction.x,
                                          y: cy + s.prediction.y)) {
                inside += 1
            }
        }
        return Double(inside) / Double(scored)
    }

    /// Effective sampling rate over the scored windows.
    var sampleRateHz: Double {
        guard scoredCellCount > 0, captureDuration > 0 else { return .nan }
        return Double(totalSampleCount)
            / (Double(scoredCellCount) * captureDuration)
    }

    private func meanOverCells(_ metric: (FixationCell) -> Double) -> Double {
        let vals = scoredCells.map(metric).filter { $0.isFinite }
        guard !vals.isEmpty else { return .nan }
        return vals.reduce(0, +) / Double(vals.count)
    }

    private func meanOverCells(_ kp: KeyPath<FixationCell, Double>) -> Double {
        meanOverCells { $0[keyPath: kp] }
    }

    private func axisSD(_ cell: FixationCell,
                        _ axis: KeyPath<simd_double2, Double>) -> Double {
        guard cell.samples.count > 1 else { return .nan }
        let vals = cell.samples.map { $0.prediction[keyPath: axis] }
        let n = Double(vals.count)
        let mean = vals.reduce(0, +) / n
        let varSum = vals.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) }
        return sqrt(varSum / (n - 1))
    }

    static let masterLogFilename = "experiment2_fixation_log.csv"
    static func masterLogURL() -> URL {
        FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(masterLogFilename)
    }

    func csvString() -> String {
        var lines: [String] = []

        // Per-cell summary first — this is the table most analyses want.
        lines.append("# Per-cell")
        lines.append("cell_idx,row,col,order,target_x_pt,target_y_pt,mean_dev_pt,mean_dev_deg,sd_pt,rms_pt,rms_deg,bias_pt,bias_deg,containment_pct,n_samples")
        for c in cells.sorted(by: { $0.index < $1.index }) {
            var row: [String] = []
            row.append("\(c.index)")
            row.append("\(c.row)")
            row.append("\(c.col)")
            row.append("\(c.order)")
            row.append(fmt(c.target.x, 2))
            row.append(fmt(c.target.y, 2))
            row.append(fmt(c.meanDeviationPoints, 2))
            row.append(fmt(VisualAngle.degrees(points: c.meanDeviationPoints,
                                               distancePoints: distancePoints), 3))
            row.append(fmt(c.standardDeviationPoints, 2))
            row.append(fmt(c.rmsPoints, 2))
            row.append(fmt(VisualAngle.degrees(points: c.rmsPoints,
                                               distancePoints: distancePoints), 3))
            row.append(fmt(c.biasPoints, 2))
            row.append(fmt(VisualAngle.degrees(points: c.biasPoints,
                                               distancePoints: distancePoints), 3))
            row.append(fmt(c.containment(screenSize: screenSize) * 100.0, 2))
            row.append("\(c.sampleCount)")
            lines.append(row.joined(separator: ","))
        }

        // Then every scored frame.
        lines.append("")
        lines.append("# Samples")
        lines.append("cell_idx,order,t_s,pred_x_pt,pred_y_pt,dev_x_pt,dev_y_pt,dev_pt,head_yaw_deg,head_pitch_deg,head_roll_deg")
        for c in cells.sorted(by: { $0.order < $1.order }) {
            for s in c.samples {
                let dx = s.prediction.x - c.target.x
                let dy = s.prediction.y - c.target.y
                var row: [String] = []
                row.append("\(c.index)")
                row.append("\(c.order)")
                row.append(fmt(s.t, 4))
                row.append(fmt(s.prediction.x, 2))
                row.append(fmt(s.prediction.y, 2))
                row.append(fmt(dx, 2))
                row.append(fmt(dy, 2))
                row.append(fmt(sqrt(dx * dx + dy * dy), 2))
                row.append(fmt(s.headYawDeg, 3))
                row.append(fmt(s.headPitchDeg, 3))
                row.append(fmt(s.headRollDeg, 3))
                lines.append(row.joined(separator: ","))
            }
        }

        lines.append("")
        lines.append("# Overall")
        lines.append("rows,cols,cells_scored,n_samples,capture_s_per_cell,sample_rate_hz,mean_dev_pt,mean_dev_deg,sd_pt,sd_deg,rms_pt,rms_deg,worst_rms_deg,bias_pt,bias_deg,sd_x_pt,sd_y_pt,containment_pct,tz_points,screen_w_pt,screen_h_pt")
        var o: [String] = []
        o.append("\(rows)")
        o.append("\(cols)")
        o.append("\(scoredCellCount)")
        o.append("\(totalSampleCount)")
        o.append(fmt(captureDuration, 2))
        o.append(fmt(sampleRateHz, 2))
        o.append(fmt(meanDeviationPoints, 2))
        o.append(fmt(meanDeviationDegrees, 3))
        o.append(fmt(standardDeviationPoints, 2))
        o.append(fmt(standardDeviationDegrees, 3))
        o.append(fmt(rmsPoints, 2))
        o.append(fmt(rmsDegrees, 3))
        o.append(fmt(worstRmsDegrees, 3))
        o.append(fmt(biasPoints, 2))
        o.append(fmt(biasDegrees, 3))
        o.append(fmt(sdX, 2))
        o.append(fmt(sdY, 2))
        o.append(fmt(containment * 100.0, 2))
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
            .integer(forKey: "experiment2_next_run_id") + 1
        UserDefaults.standard.set(runId, forKey: "experiment2_next_run_id")
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let stamp = df.string(from: Date())
        var section = "\n=== Run \(runId) @ \(stamp) — Experiment 2 (fixation stability, \(rows)×\(cols)) ===\n"
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

/// Drives one fixation-stability run: settle → capture per cell, advancing
/// through a shuffled cell order until every cell has been held once.
@MainActor
final class FixationStabilityController: ObservableObject {

    enum Phase: Equatable {
        case idle
        case settling
        case capturing
        case complete
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    /// `[0, 1]` through the current phase.
    @Published private(set) var phaseProgress: Double = 0
    /// Index into `cellOrder` of the cell being held right now.
    @Published private(set) var trialIndex: Int = 0
    /// Live scored-sample count for the active cell, so the overlay can show
    /// the run is alive without any instruction text.
    @Published private(set) var sampleCount: Int = 0
    @Published private(set) var result: FixationStabilityResult?

    let rows: Int
    let cols: Int
    let screenSize: CGSize
    let calibration: CalibrationModel
    let settleDuration: CFTimeInterval
    /// Scored seconds per cell.
    let captureDuration: CFTimeInterval
    /// Cells in row-major order.
    private(set) var cells: [FixationCell]
    /// Cell indices in presentation order (shuffled).
    let cellOrder: [Int]

    private var phaseStart: CFTimeInterval = 0
    private var captureStart: CFTimeInterval = 0
    private var timer: Timer?

    private var runLog: ExperimentRunLog
    private(set) var runBundleURL: URL?

    /// Invoked once when the run completes, on the main actor. The capture
    /// window always ends on the timer, never on a gaze frame, so the owner
    /// cannot rely on seeing `.complete` during its next `ingest` — if the
    /// face is lost as the window closes, no further frames arrive.
    var onComplete: ((FixationStabilityResult) -> Void)?

    /// - Parameters:
    ///   - captureDuration: scored seconds **per cell**. The default is short
    ///     relative to a single-box protocol because the run now holds 16
    ///     fixations: 1 s + 4 s across 16 cells is ~80 s total, and still
    ///     yields ~120 frames per cell at 30 Hz — ample for a stable RMS.
    init(screenSize: CGSize,
         calibration: CalibrationModel,
         rows: Int = 4,
         cols: Int = 4,
         settleDuration: CFTimeInterval = 1.0,
         captureDuration: CFTimeInterval = 4.0) {
        self.screenSize = screenSize
        self.calibration = calibration
        self.rows = rows
        self.cols = cols
        self.settleDuration = settleDuration
        self.captureDuration = captureDuration

        let cellW = screenSize.width / CGFloat(cols)
        let cellH = screenSize.height / CGFloat(rows)
        let halfW = Double(screenSize.width) * 0.5
        let halfH = Double(screenSize.height) * 0.5
        var built: [FixationCell] = []
        built.reserveCapacity(rows * cols)
        for idx in 0..<(rows * cols) {
            let r = idx / cols
            let c = idx % cols
            let rect = CGRect(x: CGFloat(c) * cellW,
                              y: CGFloat(r) * cellH,
                              width: cellW,
                              height: cellH)
            built.append(FixationCell(
                index: idx,
                row: r,
                col: c,
                rect: rect,
                // Centre-relative, matching the prediction space the
                // statistics are computed in.
                target: simd_double2(Double(rect.midX) - halfW,
                                     Double(rect.midY) - halfH)))
        }
        // Shuffled so any slow drift over the run isn't aliased onto screen
        // position, which would masquerade as a position-dependent effect.
        let order = Array(0..<(rows * cols)).shuffled()
        for (presentation, cellIdx) in order.enumerated() {
            built[cellIdx].order = presentation + 1
        }
        self.cells = built
        self.cellOrder = order

        self.runLog = ExperimentRunLog(
            experiment: "exp2",
            variant: "fixation_stability_\(rows)x\(cols)",
            runLabel: "Experiment 2 (fixation stability, \(rows)×\(cols))",
            // Whichever stream `GazeViewModel` is feeding us — see
            // `PipelineTuning.fixationScoredStream`. Stamped into the run so
            // no run's numbers are ambiguous: "raw" is the estimator's noise
            // floor, "filtered" is the smoothed cursor the participant sees,
            // and the two are not comparable across runs.
            predictionSource: PipelineTuning.fixationScoredStream.rawValue,
            screenSize: screenSize,
            rows: rows,
            cols: cols,
            timing: ["settle_s": settleDuration,
                     "capture_s": captureDuration])
    }

    deinit { timer?.invalidate() }

    /// The cell currently being held, or nil outside a run.
    var currentCell: FixationCell? {
        guard trialIndex < cellOrder.count else { return nil }
        return cells[cellOrder[trialIndex]]
    }

    /// Circle centre of the active cell in absolute screen points.
    var circleCenter: CGPoint { currentCell?.center ?? .zero }

    /// Active cell box in absolute screen points.
    var boxRect: CGRect { currentCell?.rect ?? .zero }

    var trialCount: Int { cellOrder.count }

    func start() {
        guard screenSize.width > 0, screenSize.height > 0 else {
            phase = .failed("No screen size")
            return
        }
        guard !cellOrder.isEmpty else {
            phase = .failed("Empty grid")
            return
        }
        timer?.invalidate()
        for i in cells.indices { cells[i].samples.removeAll() }
        trialIndex = 0
        sampleCount = 0
        phaseStart = CACurrentMediaTime()
        phaseProgress = 0
        phase = .settling
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
        phaseProgress = 0
        result = nil
    }

    /// Per-frame feed. Whether `predictionScreenAbs` is the unsmoothed
    /// projected point or the smoothed one the cursor is drawn from is
    /// `PipelineTuning.fixationScoredStream`, recorded as `predictionSource`.
    /// On `.raw`, frames with no raw sample (blink-gated) are simply not
    /// delivered here, so a held-over estimate can never enter the scatter
    /// as a zero-variance sample; on `.filtered` the blink holds are part of
    /// the stream, because they are part of what the participant sees.
    ///
    /// `diagnostics` carries the other stream alongside, so an off-device
    /// analysis can still compare the two without either being derived from
    /// the other.
    func ingest(predictionScreenAbs: CGPoint,
                gazeCam: simd_double3,
                headPose: HeadPose,
                pupilDiameters: PupilMeasure.Diameters = .init(leftPx: .nan, rightPx: .nan),
                diagnostics: ExperimentRunLog.Diagnostics = .empty) {
        guard phase == .settling || phase == .capturing else { return }
        guard let active = currentCell else { return }
        let now = CACurrentMediaTime()
        let e = headPose.euler

        runLog.append(.init(
            t: now,
            trialOrder: trialIndex + 1,
            cellIdx: active.index,
            row: active.row,
            col: active.col,
            phase: phase == .capturing ? "capture" : "settle",
            cellRect: active.rect,
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

        // Settle frames are logged (so the transient is inspectable
        // off-device) but never scored.
        guard phase == .capturing else { return }

        let rel = simd_double2(
            Double(predictionScreenAbs.x) - Double(screenSize.width) * 0.5,
            Double(predictionScreenAbs.y) - Double(screenSize.height) * 0.5)
        cells[active.index].samples.append(FixationSample(
            t: now - captureStart,
            cellIdx: active.index,
            prediction: rel,
            headYawDeg: e.yaw,
            headPitchDeg: e.pitch,
            headRollDeg: e.roll))
        sampleCount = cells[active.index].samples.count
    }

    private func tick() {
        let elapsed = CACurrentMediaTime() - phaseStart
        switch phase {
        case .settling:
            phaseProgress = max(0, min(1, elapsed / settleDuration))
            if elapsed >= settleDuration {
                phase = .capturing
                phaseStart = CACurrentMediaTime()
                captureStart = phaseStart
                phaseProgress = 0
            }
        case .capturing:
            phaseProgress = max(0, min(1, elapsed / captureDuration))
            if elapsed >= captureDuration {
                advance()
            }
        default:
            break
        }
    }

    private func advance() {
        let next = trialIndex + 1
        if next < cellOrder.count {
            trialIndex = next
            sampleCount = 0
            phaseStart = CACurrentMediaTime()
            phaseProgress = 0
            phase = .settling
        } else {
            timer?.invalidate()
            timer = nil
            finish()
        }
    }

    private func finish() {
        var r = FixationStabilityResult(
            rows: rows,
            cols: cols,
            screenSize: screenSize,
            cells: cells,
            captureDuration: captureDuration,
            distancePoints: VisualAngle.distancePoints(for: calibration))
        do {
            let out = try runLog.finish(
                trialsCSV: r.csvString(),
                summary: [
                    "rows": rows,
                    "cols": cols,
                    "cells_scored": r.scoredCellCount,
                    "n_samples": r.totalSampleCount,
                    "mean_dev_pt": r.meanDeviationPoints,
                    "mean_dev_deg": r.meanDeviationDegrees,
                    "sd_pt": r.standardDeviationPoints,
                    "rms_pt": r.rmsPoints,
                    "rms_deg": r.rmsDegrees,
                    "worst_rms_deg": r.worstRmsDegrees,
                    "bias_pt": r.biasPoints,
                    "sd_x_pt": r.sdX,
                    "sd_y_pt": r.sdY,
                    "containment_pct": r.containment * 100.0,
                    "sample_rate_hz": r.sampleRateHz,
                ])
            self.runBundleURL = out.zip
            r.runBundleURL = out.zip
            // Join this launch's session, so the operator can ship every run
            // of this experiment as one archive without hand-filtering.
            ExperimentSession.shared.record(experiment: "exp2",
                                            label: "\(rows)×\(cols)",
                                            directory: out.directory)
        } catch {
            print("experiment2 fixation run bundle failed: \(error)")
        }
        self.result = r
        phase = .complete
        onComplete?(r)
    }
}
