import Foundation
import simd
import QuartzCore
import CoreGraphics

/// One final per-dot record produced at the end of the capture window.
/// `prediction` is the mean of the window (the value used to compute the
/// displayed accuracy); head pose / raw gaze are from the last captured
/// frame in that window.
struct AccuracySample {
    let t: CFTimeInterval
    let dotIndex: Int
    let target: simd_double2
    let gazeCam: simd_double3
    let headYawDeg: Double
    let headPitchDeg: Double
    let headRollDeg: Double
    let headTx: Double
    let headTy: Double
    let headTz: Double
    let prediction: simd_double2
}

/// Result of an accuracy test run: per-target sample sets, plus aggregate
/// error metrics. Coordinates are screen-center-relative **points** (UIKit
/// logical units; +x right, +y down) to match `CalibrationModel`.
struct AccuracyTestResult {
    struct PointResult {
        let target: simd_double2
        /// Calibrated predictions captured during this target's window.
        let predictions: [simd_double2]
        /// Mean predicted point (centroid). `target - mean` is the systematic
        /// error vector for this dot.
        let mean: simd_double2
        /// RMS scatter of `predictions` around `mean` (a single scalar
        /// summary of jitter; separates random noise from systematic offset).
        let rmsScatter: Double

        var errorVector: simd_double2 { mean - target }
        var errorPoints: Double { simd_length(errorVector) }
    }

    let points: [PointResult]
    /// Calibration's fitted `|tz|` in screen points. Used as the virtual
    /// eye-to-screen distance for the points → degrees-of-visual-angle
    /// conversion below. Self-consistent with the calibration fit; note this
    /// is a virtual distance (eye position isn't in the projection — see
    /// PIPELINE.md §Stage 5 / `ScreenMapper`), not a measured physical one.
    let tzPoints: Double

    /// One final entry per dot, in dot order. Used by the CSV export.
    let samples: [AccuracySample]

    /// Mean of per-target Euclidean error magnitudes (in points).
    var meanErrorPoints: Double {
        guard !points.isEmpty else { return .nan }
        return points.map(\.errorPoints).reduce(0, +) / Double(points.count)
    }

    /// Mean angular error in degrees, using `tzPoints` as virtual distance.
    /// For each point: `atan(error_points / tz_points)`.
    var meanErrorDegrees: Double {
        guard !points.isEmpty, tzPoints > 0 else { return .nan }
        let toDeg = 180.0 / .pi
        let degs = points.map { atan($0.errorPoints / tzPoints) * toDeg }
        return degs.reduce(0, +) / Double(degs.count)
    }

    /// Build a CSV string with one row per dot (the final prediction used to
    /// compute accuracy + head pose / raw gaze from the last frame of that
    /// dot's capture window), followed by per-dot scatter and overall mean.
    /// Excel / Numbers / pandas all open this directly.
    func csvString() -> String {
        var lines: [String] = []
        lines.append([
            "dot", "target_x_pt", "target_y_pt",
            "gaze_cam_x", "gaze_cam_y", "gaze_cam_z",
            "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
            "head_tx_mm", "head_ty_mm", "head_tz_mm",
            "pred_x_pt", "pred_y_pt",
            "err_x_pt", "err_y_pt", "err_pt"
        ].joined(separator: ","))

        for s in samples {
            let err = s.prediction - s.target
            let errMag = simd_length(err)
            let row: [String] = [
                "\(s.dotIndex + 1)",
                fmt(s.target.x, 2), fmt(s.target.y, 2),
                fmt(s.gazeCam.x, 6), fmt(s.gazeCam.y, 6), fmt(s.gazeCam.z, 6),
                fmt(s.headYawDeg, 3), fmt(s.headPitchDeg, 3), fmt(s.headRollDeg, 3),
                fmt(s.headTx, 2), fmt(s.headTy, 2), fmt(s.headTz, 2),
                fmt(s.prediction.x, 2), fmt(s.prediction.y, 2),
                fmt(err.x, 2), fmt(err.y, 2), fmt(errMag, 2)
            ]
            lines.append(row.joined(separator: ","))
        }

        lines.append("")
        lines.append("# Per-dot summary")
        lines.append("dot,target_x_pt,target_y_pt,mean_x_pt,mean_y_pt,err_x_pt,err_y_pt,err_pt,rms_scatter_pt,n_samples")
        for (i, p) in points.enumerated() {
            let row: [String] = [
                "\(i + 1)",
                fmt(p.target.x, 2), fmt(p.target.y, 2),
                fmt(p.mean.x, 2), fmt(p.mean.y, 2),
                fmt(p.errorVector.x, 2), fmt(p.errorVector.y, 2),
                fmt(p.errorPoints, 2),
                p.rmsScatter.isFinite ? fmt(p.rmsScatter, 2) : "",
                "\(p.predictions.count)"
            ]
            lines.append(row.joined(separator: ","))
        }
        lines.append("")
        lines.append("# Overall")
        lines.append("mean_error_pt,mean_error_deg,tz_points")
        lines.append([
            fmt(meanErrorPoints, 2),
            fmt(meanErrorDegrees, 3),
            fmt(tzPoints, 2)
        ].joined(separator: ","))

        return lines.joined(separator: "\n") + "\n"
    }

    private func fmt(_ x: Double, _ places: Int) -> String {
        guard x.isFinite else { return "" }
        return String(format: "%.\(places)f", x)
    }

    /// Filename of the cumulative log inside the app's Documents directory.
    /// Every completed run appends its full section here.
    static let masterLogFilename = "accuracy_log.csv"

    static func masterLogURL() -> URL {
        FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(masterLogFilename)
    }

    /// Append this run's section (per-dot rows + summary blocks) to the
    /// master log file, returning the file URL. A monotonically increasing
    /// `run_id` is tracked in `UserDefaults` so runs stay numbered across
    /// app launches. Each section is preceded by a `=== Run N @ ts ===`
    /// header so individual runs are easy to spot when scanning the file.
    @discardableResult
    func appendToMasterLog() throws -> URL {
        let url = Self.masterLogURL()
        let runId = UserDefaults.standard.integer(forKey: "accuracy_next_run_id") + 1
        UserDefaults.standard.set(runId, forKey: "accuracy_next_run_id")

        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm:ss"
        let stamp = df.string(from: Date())

        var section = "\n=== Run \(runId) @ \(stamp) ===\n"
        section += csvString()

        if FileManager.default.fileExists(atPath: url.path) {
            let handle = try FileHandle(forWritingTo: url)
            defer { try? handle.close() }
            try handle.seekToEnd()
            if let data = section.data(using: .utf8) {
                try handle.write(contentsOf: data)
            }
        } else {
            // First run: drop the leading blank line so the file doesn't
            // start with whitespace.
            let first = section.hasPrefix("\n") ? String(section.dropFirst()) : section
            try first.write(to: url, atomically: true, encoding: .utf8)
        }
        return url
    }
}

/// Drives a 5-dot accuracy test: dwell → capture per dot, advance, summarize.
///
/// Mirrors `CalibrationController` (1 s dwell, 2 s capture). The user has
/// already calibrated, so each captured gaze sample is fed through
/// `CalibrationModel.predict` to get a screen-center-relative prediction;
/// the residual vs. the target is the per-frame error.
@MainActor
final class AccuracyTestController: ObservableObject {

    enum Phase: Equatable {
        case idle
        case dwelling
        case capturing
        case complete
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var dotIndex: Int = 0
    @Published private(set) var dotProgress: Double = 0
    @Published private(set) var result: AccuracyTestResult?

    let targets: [simd_double2]
    let calibration: CalibrationModel
    let preDuration: CFTimeInterval
    let captureDuration: CFTimeInterval

    private var samples: [[AccuracySample]]
    private var dotStart: CFTimeInterval = 0
    private var timer: Timer?

    init(targets: [simd_double2],
         calibration: CalibrationModel,
         preDuration: CFTimeInterval = 1.0,
         captureDuration: CFTimeInterval = 2.0) {
        self.targets = targets
        self.calibration = calibration
        self.preDuration = preDuration
        self.captureDuration = captureDuration
        self.samples = Array(repeating: [], count: targets.count)
    }

    deinit { timer?.invalidate() }

    func start() {
        guard !targets.isEmpty else {
            phase = .failed("No targets")
            return
        }
        timer?.invalidate()
        dotIndex = 0
        dotStart = CACurrentMediaTime()
        dotProgress = 0
        phase = .dwelling
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
        dotProgress = 0
        result = nil
    }

    /// Feed a raw camera-frame gaze sample plus the matching head pose.
    /// Calibrated prediction is computed internally and stored alongside the
    /// raw inputs so the CSV export is self-contained. Ignored outside the
    /// capture window.
    func ingest(gazeCam: simd_double3, headPose: HeadPose) {
        guard phase == .capturing else { return }
        guard dotIndex < samples.count else { return }
        let p = calibration.predict(gazeCam: gazeCam)
        guard p.x.isFinite, p.y.isFinite else { return }
        let euler = headPose.euler
        let s = AccuracySample(
            t: CACurrentMediaTime(),
            dotIndex: dotIndex,
            target: targets[dotIndex],
            gazeCam: gazeCam,
            headYawDeg: euler.yaw,
            headPitchDeg: euler.pitch,
            headRollDeg: euler.roll,
            headTx: headPose.translation.x,
            headTy: headPose.translation.y,
            headTz: headPose.translation.z,
            prediction: p
        )
        samples[dotIndex].append(s)
    }

    var currentTarget: simd_double2 {
        guard dotIndex < targets.count else { return .zero }
        return targets[dotIndex]
    }

    private func tick() {
        let elapsed = CACurrentMediaTime() - dotStart
        let total = preDuration + captureDuration
        dotProgress = min(1.0, elapsed / total)
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
        let next = dotIndex + 1
        if next < targets.count {
            dotIndex = next
            dotStart = CACurrentMediaTime()
            dotProgress = 0
            phase = .dwelling
        } else {
            timer?.invalidate()
            timer = nil
            finish()
        }
    }

    private func finish() {
        var pointResults: [AccuracyTestResult.PointResult] = []
        pointResults.reserveCapacity(targets.count)
        var finalSamples: [AccuracySample] = []
        var totalCaptured = 0
        for (i, ds) in samples.enumerated() {
            totalCaptured += ds.count
            guard !ds.isEmpty else {
                pointResults.append(.init(
                    target: targets[i],
                    predictions: [],
                    mean: targets[i],
                    rmsScatter: .nan))
                continue
            }
            let preds = ds.map(\.prediction)
            let n = Double(preds.count)
            let mean = preds.reduce(simd_double2(0, 0), +) / n
            let ss = preds.reduce(0.0) { acc, p in
                let d = p - mean
                return acc + simd_dot(d, d)
            }
            let rms = sqrt(ss / n)
            pointResults.append(.init(
                target: targets[i],
                predictions: preds,
                mean: mean,
                rmsScatter: rms))

            // One representative "final" sample per dot: head pose / raw gaze
            // from the last captured frame, prediction overridden to the mean
            // used for the accuracy calculation.
            let last = ds.last!
            finalSamples.append(AccuracySample(
                t: last.t,
                dotIndex: last.dotIndex,
                target: last.target,
                gazeCam: last.gazeCam,
                headYawDeg: last.headYawDeg,
                headPitchDeg: last.headPitchDeg,
                headRollDeg: last.headRollDeg,
                headTx: last.headTx,
                headTy: last.headTy,
                headTz: last.headTz,
                prediction: mean
            ))
        }
        guard totalCaptured > 0 else {
            phase = .failed("No gaze samples captured")
            return
        }
        self.result = AccuracyTestResult(
            points: pointResults,
            tzPoints: abs(calibration.translation.z),
            samples: finalSamples
        )
        phase = .complete
    }

    /// 5-dot cross: center + 4 corners. Same 80% inset as `standardTargets`
    /// so corners stay clear of the safe-area edges.
    static func crossTargets(screenSize: CGSize) -> [simd_double2] {
        let halfW = Double(screenSize.width)  * 0.40
        let halfH = Double(screenSize.height) * 0.40
        return [
            simd_double2(0, 0),
            simd_double2(-halfW, -halfH),
            simd_double2( halfW, -halfH),
            simd_double2(-halfW,  halfH),
            simd_double2( halfW,  halfH),
        ]
    }
}
