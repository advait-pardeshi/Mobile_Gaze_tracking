import Foundation
import simd
import QuartzCore
import CoreGraphics

/// Per-run experiment bundle, shared by the three experiments.
///
/// Each run gets its own directory under
/// `Documents/experiment_runs/<exp>_<variant>_run<N>_<stamp>/` holding:
///
///   * `samples.csv` — one row per ingested camera frame for the WHOLE run
///     (Exp 1's dwell phase included, hit and miss cells alike).
///   * `trials.csv`  — one row per trial; the same body the experiment
///     appends to its cumulative master log, this run only.
///   * `meta.json`   — run identity, grid/screen/timing parameters and the
///     summary stats, so a bundle is self-describing off-device.
///
/// The directory is zipped on `finish()` and the archive URL handed to the
/// results view for the share sheet.
///
/// Sample rows are screen-absolute points so analysis doesn't need the
/// screen size (it's in `meta.json` anyway). Rows are buffered in memory
/// and flushed once at `finish()`: `append` runs on the main actor inside
/// the gaze path, where a per-frame disk write would show up as jitter.
/// A 9×9 run is roughly 15 k rows, which is a few MB of text — fine to
/// hold.
struct ExperimentRunLog {

    /// Bumped when the `samples.csv` column set or the `meta.json` shape
    /// changes, so off-device tooling can branch on it.
    ///
    /// v2 dropped the eight upstream-smoother / facing-deviation columns
    /// (`filt_*`, `eye_cam_*_f_mm`, `head_dev_*`, `upstream_smoothing`).
    /// This build has no `GazeUpstreamSmoother`, so nothing ever wrote
    /// them and every row carried eight empty cells.
    /// v3 dropped `gaze_cam_x/y/z` from `samples.csv`. Note this is lossy:
    /// the camera-frame gaze direction cannot be recovered from
    /// `raw_pitch_deg`/`raw_yaw_deg` without the normalization rotation
    /// `R_n`, which is not logged. `trials.csv` still carries its own
    /// per-trial `gaze_cam_*` snapshot.
    static let schemaVersion = 3

    /// Per-frame diagnostic bundle for noise attribution: the same frame
    /// observed at several points in the pipeline, so an off-device
    /// analysis can separate *model* noise from *pose* noise from *filter*
    /// lag instead of guessing which one costs the most.
    ///
    /// Suggested analysis on a held fixation:
    ///   * `std(raw_pitch_deg, raw_yaw_deg)` → CNN + normalization noise.
    ///   * `std(eye_cam_z_mm)` → PnP depth noise, the term multiplied into
    ///     the screen point by `μ = -tz/gz` in `ScreenMapper.planeIntersect`.
    ///   * `std(raw_pred_*)` vs `std(pred_*)` → what the Kalman filter buys.
    ///   * `count(|Δ raw_pred| > 60 pt)` → outlier rate at the source.
    ///
    /// All fields default to NaN so call sites that don't have a value
    /// write an empty CSV cell rather than a misleading zero.
    struct Diagnostics {
        /// Raw CNN output angles, degrees, before any smoothing.
        var rawPitchDeg: Double = .nan
        var rawYawDeg: Double = .nan
        /// Screen point from the *unfiltered* gaze + eye position, absolute
        /// screen points. The `pred_x/pred_y` columns hold whichever point
        /// the experiment actually acts on (see `predictionSource`).
        var rawPredX: Double = .nan
        var rawPredY: Double = .nan
        /// PnP eye midpoint (gaze-ray origin), camera frame, mm — raw.
        /// Written to the `eye_cam_*_mm` columns.
        var eyeCam = simd_double3(.nan, .nan, .nan)

        static let empty = Diagnostics()
    }

    /// One ingested camera frame.
    ///
    /// `target_*`, `err_pt` and `in_cell` are derived from `cellRect` rather
    /// than passed in, so a sample row can never disagree with the rect the
    /// experiment is actually hit-testing against.
    struct Sample {
        /// `CACurrentMediaTime()` at capture; written relative to run start.
        let t: CFTimeInterval
        /// 1-based presentation order of the active trial.
        let trialOrder: Int
        let cellIdx: Int
        let row: Int
        let col: Int
        /// "dwell"/"capture" for Exp 1, "active" for Exp 2/3.
        let phase: String
        /// Active cell rect, absolute screen points.
        let cellRect: CGRect
        /// Predicted gaze point, absolute screen points.
        let predX: Double
        let predY: Double
        let headYawDeg: Double
        let headPitchDeg: Double
        let headRollDeg: Double
        let headTx: Double
        let headTy: Double
        let headTz: Double
        /// Iris-diameter proxy in source-image pixels; NaN when unmeasured.
        var pupilLeftPx: Double = .nan
        var pupilRightPx: Double = .nan
        /// Pipeline-stage diagnostics; defaults to all-NaN when unset.
        var diag: Diagnostics = .empty

        var targetX: Double { Double(cellRect.midX) }
        var targetY: Double { Double(cellRect.midY) }

        /// Euclidean distance from this frame's prediction to the cell
        /// center, in points.
        var errPt: Double {
            let dx = predX - targetX
            let dy = predY - targetY
            return sqrt(dx * dx + dy * dy)
        }

        var inCell: Bool {
            cellRect.contains(CGPoint(x: predX, y: predY))
        }
    }

    /// Column order of `samples.csv`. The first 23 are the analysis set;
    /// the last 4 are `Diagnostics` (see above).
    static let sampleHeader: [String] = [
        "t_s", "trial", "cell_idx", "row", "col",
        "target_x", "target_y",
        "phase",
        "pred_x", "pred_y", "err_pt", "in_cell",
        "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
        "head_tx_mm", "head_ty_mm", "head_tz_mm",
        "eye_cam_x_mm", "eye_cam_y_mm", "eye_cam_z_mm",
        "pupil_left_px", "pupil_right_px",
        // Diagnostics — see `ExperimentRunLog.Diagnostics`.
        "raw_pitch_deg", "raw_yaw_deg",
        "raw_pred_x", "raw_pred_y",
    ]

    /// Short experiment key: "exp1" / "exp2" / "exp3".
    let experiment: String
    /// Filename-safe variant key, e.g. "9x9_walking", "5x4_images".
    let variant: String
    /// Human-readable run description, echoed into `meta.json`.
    let runLabel: String
    /// Which prediction the `pred_x/pred_y` columns hold. Exp 1 logs the
    /// raw calibrated point (what its trial mean is built from); Exp 2/3
    /// log the smoothed point (what hit detection uses).
    let predictionSource: String
    let screenSize: CGSize
    let rows: Int
    let cols: Int
    /// Experiment timing parameters, verbatim into `meta.json`.
    let timing: [String: Double]

    private(set) var samples: [Sample] = []
    private var runStart: CFTimeInterval = 0
    private var startedAt = Date()

    init(experiment: String,
         variant: String,
         runLabel: String,
         predictionSource: String,
         screenSize: CGSize,
         rows: Int,
         cols: Int,
         timing: [String: Double] = [:]) {
        self.experiment = experiment
        self.variant = variant
        self.runLabel = runLabel
        self.predictionSource = predictionSource
        self.screenSize = screenSize
        self.rows = rows
        self.cols = cols
        self.timing = timing
    }

    mutating func begin() {
        samples.removeAll()
        runStart = CACurrentMediaTime()
        startedAt = Date()
    }

    mutating func append(_ s: Sample) {
        samples.append(s)
    }

    /// Parent directory of every run bundle.
    static func runsRoot() -> URL {
        FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("experiment_runs", isDirectory: true)
    }

    func samplesCSV() -> String {
        var lines: [String] = [Self.sampleHeader.joined(separator: ",")]
        lines.reserveCapacity(samples.count + 1)
        for s in samples {
            lines.append([
                fmt(s.t - runStart, 4),
                "\(s.trialOrder)", "\(s.cellIdx)", "\(s.row)", "\(s.col)",
                fmt(s.targetX, 2), fmt(s.targetY, 2),
                s.phase,
                fmt(s.predX, 2), fmt(s.predY, 2),
                fmt(s.errPt, 2), s.inCell ? "1" : "0",
                fmt(s.headYawDeg, 3), fmt(s.headPitchDeg, 3), fmt(s.headRollDeg, 3),
                fmt(s.headTx, 2), fmt(s.headTy, 2), fmt(s.headTz, 2),
                fmt(s.diag.eyeCam.x, 2), fmt(s.diag.eyeCam.y, 2),
                fmt(s.diag.eyeCam.z, 2),
                fmt(s.pupilLeftPx, 3), fmt(s.pupilRightPx, 3),
                fmt(s.diag.rawPitchDeg, 4), fmt(s.diag.rawYawDeg, 4),
                fmt(s.diag.rawPredX, 2), fmt(s.diag.rawPredY, 2),
            ].joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// Write the bundle and zip it.
    ///
    /// `trialsCSV` is the experiment's own per-trial table (the master-log
    /// body for this run, footer included or not — the caller decides).
    /// `summary` is merged into `meta.json` under `summary`.
    ///
    /// Returns the run directory and, when zipping succeeded, the archive.
    /// A failed zip is not fatal: the directory is already on disk and
    /// reachable over Files sharing, so the run is never lost to it.
    @discardableResult
    func finish(trialsCSV: String,
                summary: [String: Any]) throws -> (directory: URL, zip: URL?) {
        let key = "\(experiment)_run_next_id"
        let runId = UserDefaults.standard.integer(forKey: key) + 1
        UserDefaults.standard.set(runId, forKey: key)

        let stampFmt = DateFormatter()
        stampFmt.dateFormat = "yyyyMMdd_HHmmss"
        let stamp = stampFmt.string(from: startedAt)

        let name = "\(experiment)_\(variant)_run\(runId)_\(stamp)"
        let dir = Self.runsRoot().appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: dir,
                                                withIntermediateDirectories: true)

        try samplesCSV().write(to: dir.appendingPathComponent("samples.csv"),
                               atomically: true, encoding: .utf8)
        try trialsCSV.write(to: dir.appendingPathComponent("trials.csv"),
                            atomically: true, encoding: .utf8)

        let isoFmt = ISO8601DateFormatter()
        var meta: [String: Any] = [
            "schema_version": Self.schemaVersion,
            "experiment": experiment,
            "variant": variant,
            "run_label": runLabel,
            "run_id": runId,
            "started_at": isoFmt.string(from: startedAt),
            "duration_s": samples.last.map { $0.t - runStart } ?? 0,
            "sample_count": samples.count,
            "prediction_source": predictionSource,
            "grid": ["rows": rows, "cols": cols],
            "screen": ["w_pt": Double(screenSize.width),
                       "h_pt": Double(screenSize.height)],
            "summary": summary,
        ]
        if !timing.isEmpty { meta["timing"] = timing }
        let metaData = try JSONSerialization.data(withJSONObject: Self.jsonSafe(meta),
                                                  options: [.prettyPrinted,
                                                            .sortedKeys])
        try metaData.write(to: dir.appendingPathComponent("meta.json"),
                           options: .atomic)

        let zipURL = Self.runsRoot().appendingPathComponent("\(name).zip")
        let zipped = DirectoryZip.zip(directory: dir, to: zipURL,
                                      tag: "ExperimentRunLog")
        return (dir, zipped ? zipURL : nil)
    }

    private func fmt(_ x: Double, _ places: Int) -> String {
        guard x.isFinite else { return "" }
        return String(format: "%.\(places)f", x)
    }

    /// `JSONSerialization` throws on NaN/±∞, and summary stats are NaN for
    /// an empty or all-miss run (`meanErrorPoints`, `meanTimeToHit`). Map
    /// those to JSON `null` so a degenerate run still writes a bundle
    /// instead of losing its samples to an exception.
    private static func jsonSafe(_ value: Any) -> Any {
        switch value {
        case let d as Double:
            return d.isFinite ? d : NSNull()
        case let dict as [String: Any]:
            return dict.mapValues { jsonSafe($0) }
        case let arr as [Any]:
            return arr.map { jsonSafe($0) }
        default:
            return value
        }
    }
}
