import Foundation
import simd
import QuartzCore
import CoreGraphics

/// Calibration validation — a free-order check of a freshly fitted
/// calibration, run immediately after `CalibrationController` completes.
///
/// Unlike calibration (and unlike the old sequential accuracy test), **all 9
/// dots are on screen at once** and the participant may look at them in any
/// order. Nothing tells them where to look next; the run simply watches which
/// dot the prediction settles on and scores that dot when it has been held
/// long enough.
///
/// Why this exists: a calibration fit always *succeeds* numerically — the
/// least-squares solve returns a `t` whether or not the participant was
/// actually fixating the dots. Validation is the step that decides whether the
/// fit is good enough to run an experiment on, instead of assuming it is.
///
/// **Dot attribution.** Each frame the prediction is assigned to the nearest
/// *unconfirmed* dot. A dot is confirmed once the prediction has stayed its
/// nearest dot, and within `captureRadius` of it, continuously for
/// `dwellRequirement` seconds. Leaving the radius — or a different dot
/// becoming nearest — restarts that dot's dwell window. Only samples from
/// inside the satisfied dwell window are scored, so the saccade *into* the
/// dot never pollutes its error estimate.
///
/// The radius gate is deliberately generous (a fraction of the inter-dot
/// spacing, not a tight tolerance): the job here is to *measure* error, not to
/// require that it be small. A gate tight enough to guarantee correct
/// attribution would make a poorly calibrated session unvalidatable, which
/// is exactly the session we most need a number for.

/// Per-dot outcome of a validation run. Geometry is screen-center-relative
/// points, matching `CalibrationModel.targets`.
struct ValidationDot: Identifiable {
    let index: Int
    /// Dot position, screen-center-relative points.
    let target: simd_double2

    /// Nil until the dot is confirmed (participant held it long enough).
    var scatter: GazeScatter?
    /// Order in which this dot was confirmed, 1-based. 0 = never confirmed.
    var confirmedOrder: Int = 0
    /// Seconds from run start to confirmation. NaN if never confirmed.
    var confirmedAtSeconds: Double = .nan

    var id: Int { index }
    var isConfirmed: Bool { scatter != nil }

    /// Systematic error vector `mean - target`, screen points.
    var errorVector: simd_double2 {
        guard let s = scatter else { return simd_double2(.nan, .nan) }
        return s.mean - target
    }
    /// Magnitude of the systematic error, screen points. This is what the
    /// per-dot accuracy figure reports.
    var errorPoints: Double { scatter?.bias ?? .nan }
    /// Precision: RMS scatter about the dot's own centroid, screen points.
    var precisionPoints: Double { scatter?.rms ?? .nan }
    var sampleCount: Int { scatter?.count ?? 0 }
}

/// Aggregate verdict of a validation run — the reason the feature exists.
///
/// Thresholds are expressed in degrees of visual angle so they don't drift
/// with screen size. They're deliberately lenient: this is a smartphone
/// front-camera appearance-based estimator, not a tower-mounted IR tracker,
/// and the point of the gate is to catch *broken* calibrations (participant
/// looked away, model not loaded, fit collapsed) rather than to certify
/// research-grade precision.
enum ValidationVerdict: String {
    case good
    case marginal
    case poor
    case incomplete

    var label: String {
        switch self {
        case .good:       return "Calibration looks good"
        case .marginal:   return "Calibration is marginal"
        case .poor:       return "Calibration is poor — re-calibrate"
        case .incomplete: return "Incomplete — not all dots confirmed"
        }
    }

    /// Short guidance shown under the verdict on the results screen.
    var advice: String {
        switch self {
        case .good:
            return "Safe to proceed to an experiment."
        case .marginal:
            return "Usable, but expect misses on fine grids (5×5)."
        case .poor:
            return "Re-calibrate before running an experiment — results would not be meaningful."
        case .incomplete:
            return "Some dots were never fixated long enough to score. Re-run the validation."
        }
    }

    static let goodMaxDegrees = 2.0
    static let marginalMaxDegrees = 4.0
    /// A run must confirm at least this many of the 9 dots to be scored at
    /// all. Below it, the fit hasn't been sampled broadly enough to trust
    /// any aggregate number.
    static let minConfirmedDots = 8
}

struct CalibrationValidationResult {
    let dots: [ValidationDot]
    let screenSize: CGSize
    /// Virtual eye-to-screen distance in points, from the calibration under
    /// test. Used for every degrees conversion here.
    let distancePoints: Double
    /// Seconds the participant took to confirm all the dots they confirmed.
    let durationSeconds: Double
    /// Zipped run bundle, nil if writing or zipping failed.
    var runBundleURL: URL? = nil

    var confirmedDots: [ValidationDot] { dots.filter(\.isConfirmed) }
    var confirmedCount: Int { confirmedDots.count }

    /// Mean systematic error across confirmed dots, screen points.
    var meanErrorPoints: Double {
        let errs = confirmedDots.map(\.errorPoints).filter { $0.isFinite }
        guard !errs.isEmpty else { return .nan }
        return errs.reduce(0, +) / Double(errs.count)
    }

    /// Mean systematic error across confirmed dots, degrees of visual angle.
    var meanErrorDegrees: Double {
        VisualAngle.meanDegrees(
            pointErrors: confirmedDots.map(\.errorPoints),
            distancePoints: distancePoints)
    }

    /// Worst single dot, degrees. A fit can look fine on average and still be
    /// unusable in one corner, which is precisely what a grid experiment
    /// would then trip over.
    var worstErrorDegrees: Double {
        let errs = confirmedDots
            .map { VisualAngle.degrees(points: $0.errorPoints,
                                       distancePoints: distancePoints) }
            .filter { $0.isFinite }
        return errs.max() ?? .nan
    }

    /// Mean precision (RMS scatter) across confirmed dots, screen points.
    var meanPrecisionPoints: Double {
        let ps = confirmedDots.map(\.precisionPoints).filter { $0.isFinite }
        guard !ps.isEmpty else { return .nan }
        return ps.reduce(0, +) / Double(ps.count)
    }

    var meanPrecisionDegrees: Double {
        VisualAngle.meanDegrees(
            pointErrors: confirmedDots.map(\.precisionPoints),
            distancePoints: distancePoints)
    }

    var verdict: ValidationVerdict {
        guard confirmedCount >= ValidationVerdict.minConfirmedDots else {
            return .incomplete
        }
        let mean = meanErrorDegrees
        guard mean.isFinite else { return .incomplete }
        if mean <= ValidationVerdict.goodMaxDegrees { return .good }
        if mean <= ValidationVerdict.marginalMaxDegrees { return .marginal }
        return .poor
    }

    func csvString() -> String {
        var lines: [String] = []
        lines.append([
            "dot", "target_x_pt", "target_y_pt",
            "confirmed", "confirm_order", "confirmed_at_s",
            "mean_x_pt", "mean_y_pt",
            "err_x_pt", "err_y_pt", "err_pt", "err_deg",
            "mean_dev_pt", "sd_pt", "rms_pt", "rms_deg",
            "n_samples"
        ].joined(separator: ","))
        // Rows are assembled by successive `append` rather than as one large
        // array literal: a 17-element heterogeneous literal mixing string
        // interpolation, optional `map` and function calls pushes Swift's
        // expression type-checker into an exponential blowup ("unable to
        // type-check in reasonable time").
        for d in dots {
            let s = d.scatter
            var row: [String] = []
            row.append("\(d.index + 1)")
            row.append(fmt(d.target.x, 2))
            row.append(fmt(d.target.y, 2))
            row.append(d.isConfirmed ? "1" : "0")
            row.append("\(d.confirmedOrder)")
            row.append(fmt(d.confirmedAtSeconds, 3))
            row.append(s.map { fmt($0.mean.x, 2) } ?? "")
            row.append(s.map { fmt($0.mean.y, 2) } ?? "")
            row.append(fmt(d.errorVector.x, 2))
            row.append(fmt(d.errorVector.y, 2))
            row.append(fmt(d.errorPoints, 2))
            row.append(fmt(VisualAngle.degrees(points: d.errorPoints,
                                               distancePoints: distancePoints), 3))
            row.append(s.map { fmt($0.meanDeviation, 2) } ?? "")
            row.append(s.map { fmt($0.standardDeviation, 2) } ?? "")
            row.append(s.map { fmt($0.rms, 2) } ?? "")
            row.append(fmt(VisualAngle.degrees(points: d.precisionPoints,
                                               distancePoints: distancePoints), 3))
            row.append("\(d.sampleCount)")
            lines.append(row.joined(separator: ","))
        }
        lines.append("")
        lines.append("# Overall")
        lines.append("confirmed_dots,total_dots,mean_err_pt,mean_err_deg,worst_err_deg,mean_rms_pt,mean_rms_deg,verdict,duration_s,tz_points,screen_w_pt,screen_h_pt")
        var overall: [String] = []
        overall.append("\(confirmedCount)")
        overall.append("\(dots.count)")
        overall.append(fmt(meanErrorPoints, 2))
        overall.append(fmt(meanErrorDegrees, 3))
        overall.append(fmt(worstErrorDegrees, 3))
        overall.append(fmt(meanPrecisionPoints, 2))
        overall.append(fmt(meanPrecisionDegrees, 3))
        overall.append(verdict.rawValue)
        overall.append(fmt(durationSeconds, 2))
        overall.append(fmt(distancePoints, 2))
        overall.append(fmt(Double(screenSize.width), 2))
        overall.append(fmt(Double(screenSize.height), 2))
        lines.append(overall.joined(separator: ","))
        return lines.joined(separator: "\n") + "\n"
    }

    private func fmt(_ x: Double, _ places: Int) -> String {
        guard x.isFinite else { return "" }
        return String(format: "%.\(places)f", x)
    }
}

/// Drives a validation run: 9 dots visible at once, confirmed in whatever
/// order the participant chooses.
@MainActor
final class CalibrationValidationController: ObservableObject {

    enum Phase: Equatable {
        case idle
        case running
        case complete
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var dots: [ValidationDot]
    /// Index of the dot currently accumulating dwell, or nil when the gaze
    /// isn't settled on any unconfirmed dot. Drives the overlay's highlight.
    @Published private(set) var activeDotIndex: Int?
    /// `[0, 1]` dwell accumulated on `activeDotIndex`.
    @Published private(set) var dwellProgress: Double = 0
    /// `[0, 1]` of the whole-run time budget consumed.
    @Published private(set) var runProgress: Double = 0
    @Published private(set) var result: CalibrationValidationResult?

    let screenSize: CGSize
    let calibration: CalibrationModel
    /// Continuous seconds the prediction must hold a dot to confirm it.
    let dwellRequirement: CFTimeInterval
    /// Max radius from a dot, in points, for the prediction to count as
    /// settled on it.
    let captureRadius: Double
    /// Whole-run budget. On expiry the run finishes with whatever dots were
    /// confirmed, and the verdict reports `.incomplete` if too few were.
    let runTimeout: CFTimeInterval

    private var runStart: CFTimeInterval = 0
    /// Timestamp the current dwell window opened, and which dot it's on.
    private var dwellDot: Int?
    private var dwellStart: CFTimeInterval = 0
    /// Samples collected during the open dwell window, center-relative.
    private var dwellSamples: [simd_double2] = []
    private var nextConfirmOrder = 1
    private var timer: Timer?

    private var runLog: ExperimentRunLog
    private(set) var runBundleURL: URL?

    /// Invoked once when the run completes, on the main actor.
    ///
    /// A run can end from the whole-run timer or the Finish button, i.e. off
    /// the gaze path — so the owner cannot rely on noticing `.complete` during
    /// its next `ingest`. If the face is lost at that moment no further frames
    /// arrive and the results screen would never appear.
    var onComplete: ((CalibrationValidationResult) -> Void)?

    init(calibration: CalibrationModel,
         screenSize: CGSize,
         dwellRequirement: CFTimeInterval = 1.0,
         runTimeout: CFTimeInterval = 120.0) {
        self.calibration = calibration
        self.screenSize = screenSize
        self.dwellRequirement = dwellRequirement
        self.runTimeout = runTimeout
        self.dots = calibration.targets.enumerated().map {
            ValidationDot(index: $0.offset, target: $0.element)
        }
        // Attribution radius from the actual dot layout rather than a fixed
        // constant, so it scales with the device. `standardTargets` spaces
        // the 3×3 grid at 40% of each screen dimension; 60% of the smaller
        // half-spacing keeps neighbouring dots from stealing each other's
        // samples while still tolerating a few degrees of error.
        let spacingX = Double(screenSize.width) * 0.40
        let spacingY = Double(screenSize.height) * 0.40
        self.captureRadius = max(40.0, min(spacingX, spacingY) * 0.6)
        self.runLog = ExperimentRunLog(
            experiment: "validation",
            variant: "9dot_free_order",
            runLabel: "Calibration validation (9 dots, free order)",
            // Attribution consumes the smoothed point drawn on screen.
            predictionSource: "smoothed",
            screenSize: screenSize,
            rows: 3,
            cols: 3,
            timing: ["dwell_requirement_s": dwellRequirement,
                     "run_timeout_s": runTimeout])
    }

    deinit { timer?.invalidate() }

    func start() {
        guard !dots.isEmpty else {
            phase = .failed("No calibration targets")
            return
        }
        guard screenSize.width > 0, screenSize.height > 0 else {
            phase = .failed("No screen size")
            return
        }
        timer?.invalidate()
        runStart = CACurrentMediaTime()
        dwellDot = nil
        dwellSamples.removeAll()
        dwellProgress = 0
        runProgress = 0
        activeDotIndex = nil
        nextConfirmOrder = 1
        phase = .running
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
        activeDotIndex = nil
        dwellProgress = 0
        result = nil
    }

    /// Finish early with whatever has been confirmed so far. Wired to the
    /// overlay's "Finish" affordance so a participant who can't hold one
    /// stubborn dot isn't stuck waiting out the full timeout.
    func finishNow() {
        guard phase == .running else { return }
        timer?.invalidate()
        timer = nil
        finish()
    }

    /// Absolute-screen centre of a dot, for the overlay.
    func absoluteCenter(of dot: ValidationDot) -> CGPoint {
        CGPoint(x: screenSize.width * 0.5 + CGFloat(dot.target.x),
                y: screenSize.height * 0.5 + CGFloat(dot.target.y))
    }

    /// Per-frame feed. `predictionScreenAbs` is the Kalman-smoothed
    /// post-calibration point — the same one drawn on screen — so what the
    /// participant sees and what gets scored are identical.
    func ingest(predictionScreenAbs: CGPoint,
                gazeCam: simd_double3,
                headPose: HeadPose,
                pupilDiameters: PupilMeasure.Diameters = .init(leftPx: .nan, rightPx: .nan),
                diagnostics: ExperimentRunLog.Diagnostics = .empty) {
        guard phase == .running else { return }

        // Center-relative, matching the dot targets.
        let rel = simd_double2(
            Double(predictionScreenAbs.x) - Double(screenSize.width) * 0.5,
            Double(predictionScreenAbs.y) - Double(screenSize.height) * 0.5)

        // Nearest dot that hasn't been confirmed yet.
        var nearest: Int?
        var nearestDist = Double.infinity
        for d in dots where !d.isConfirmed {
            let dist = simd_distance(d.target, rel)
            if dist < nearestDist { nearestDist = dist; nearest = d.index }
        }

        let now = CACurrentMediaTime()
        let settled = nearest != nil && nearestDist <= captureRadius

        // Log every frame, tagged with the dot it was attributed to (or -1
        // when the gaze was between dots), so the attribution itself can be
        // audited off-device.
        let logIdx = settled ? (nearest ?? -1) : -1
        let logRect: CGRect
        if settled, let n = nearest {
            let c = absoluteCenter(of: dots[n])
            logRect = CGRect(x: c.x - CGFloat(captureRadius),
                             y: c.y - CGFloat(captureRadius),
                             width: CGFloat(captureRadius) * 2,
                             height: CGFloat(captureRadius) * 2)
        } else {
            logRect = .null
        }
        let e = headPose.euler
        runLog.append(.init(
            t: now,
            trialOrder: nextConfirmOrder,
            cellIdx: logIdx,
            row: logIdx >= 0 ? logIdx / 3 : -1,
            col: logIdx >= 0 ? logIdx % 3 : -1,
            phase: settled ? "settled" : "searching",
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

        guard settled, let idx = nearest else {
            // Gaze is between dots — abandon any open dwell window.
            resetDwell()
            return
        }

        if dwellDot != idx {
            // Newly settled on this dot: open a fresh window. Anything
            // collected on the previous dot is discarded, so a sweep across
            // the screen can't half-fill several dots at once.
            dwellDot = idx
            dwellStart = now
            dwellSamples = [rel]
            activeDotIndex = idx
            dwellProgress = 0
            return
        }

        dwellSamples.append(rel)
        let held = now - dwellStart
        dwellProgress = max(0, min(1, held / dwellRequirement))
        if held >= dwellRequirement {
            confirm(idx, at: now)
        }
    }

    private func resetDwell() {
        if dwellDot != nil {
            dwellDot = nil
            dwellSamples.removeAll()
            dwellProgress = 0
            activeDotIndex = nil
        }
    }

    private func confirm(_ idx: Int, at now: CFTimeInterval) {
        // Score the dot from the dwell window's samples only.
        if let s = GazeScatter.compute(samples: dwellSamples,
                                       target: dots[idx].target) {
            dots[idx].scatter = s
        }
        dots[idx].confirmedOrder = nextConfirmOrder
        dots[idx].confirmedAtSeconds = now - runStart
        nextConfirmOrder += 1
        resetDwell()

        if dots.allSatisfy(\.isConfirmed) {
            timer?.invalidate()
            timer = nil
            finish()
        }
    }

    private func tick() {
        guard phase == .running else { return }
        let elapsed = CACurrentMediaTime() - runStart
        runProgress = max(0, min(1, elapsed / runTimeout))
        if elapsed >= runTimeout {
            timer?.invalidate()
            timer = nil
            finish()
        }
    }

    private func finish() {
        var r = CalibrationValidationResult(
            dots: dots,
            screenSize: screenSize,
            distancePoints: VisualAngle.distancePoints(for: calibration),
            durationSeconds: CACurrentMediaTime() - runStart)
        do {
            let out = try runLog.finish(
                trialsCSV: r.csvString(),
                summary: [
                    "confirmed_dots": r.confirmedCount,
                    "total_dots": r.dots.count,
                    "mean_err_pt": r.meanErrorPoints,
                    "mean_err_deg": r.meanErrorDegrees,
                    "worst_err_deg": r.worstErrorDegrees,
                    "mean_rms_pt": r.meanPrecisionPoints,
                    "mean_rms_deg": r.meanPrecisionDegrees,
                    "verdict": r.verdict.rawValue,
                ])
            self.runBundleURL = out.zip
            r.runBundleURL = out.zip
        } catch {
            print("validation run bundle failed: \(error)")
        }
        self.result = r
        self.activeDotIndex = nil
        phase = .complete
        onComplete?(r)
    }
}
