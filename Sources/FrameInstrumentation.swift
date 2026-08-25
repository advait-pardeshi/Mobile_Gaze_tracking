import Foundation
import simd
import QuartzCore

/// Temporary per-frame instrumentation for the `perf-and-stability` branch.
///
/// Writes one CSV row per *processed* camera frame to
/// `Documents/instrumentation/frames_<stamp>.csv`, covering four questions
/// that the experiment bundles cannot answer:
///
///  1. **Where does the frame time go** — wall-clock per pipeline stage.
///  2. **How noisy is the head pose itself** — `R_h` as a quaternion, raw and
///     filtered, plus the eye midpoint. Logged as a quaternion rather than
///     Euler angles so an off-device angular variance is wrap-free and
///     doesn't inherit the gimbal behaviour of the `HeadPose.euler`
///     decomposition.
///  3. **How deep is the queue** — frame arrival vs. publish timestamps, and
///     how many frames backpressure dropped in between.
///  4. **Which frames were blinks** — the eye-aspect-ratio behind the gate.
///
/// Rows are buffered and flushed on a background queue: this is measuring
/// per-frame latency, so it must not add a disk write to the path it
/// measures.
final class FrameInstrumentation {

    struct Row {
        var timestampMs: Int = 0
        /// `CACurrentMediaTime()` when the landmark callback fired.
        var arrival: CFTimeInterval = 0
        /// `CACurrentMediaTime()` after the main-actor publish completed.
        var publish: CFTimeInterval = 0
        /// Frames backpressure dropped since the previous processed frame.
        var droppedSince: Int = 0

        // Per-stage durations, seconds.
        var dtPose: Double = .nan
        var dtHeadFilter: Double = .nan
        var dtEyeWarp: Double = .nan
        var dtFaceWarp: Double = .nan
        var dtCNN: Double = .nan
        var dtGazeFilter: Double = .nan
        /// Queue hop: landmark callback → first line of worker processing.
        var dtDispatch: Double = .nan
        /// Worker finish → main-actor publish complete (projection included).
        var dtPublish: Double = .nan

        var poseOK: Bool = false
        var faceCropOK: Bool = false

        /// Raw PnP head rotation, quaternion (x, y, z, w).
        var qRaw: simd_quatd?
        /// Post-One-Euro head rotation actually used for the crop.
        var qFiltered: simd_quatd?
        /// Eye midpoint, camera frame, mm — raw and filtered.
        var eyeRaw: simd_double3?
        var eyeFiltered: simd_double3?

        var earLeft: Double = .nan
        var earRight: Double = .nan
        /// Running open-eye EAR the blink gate is comparing against this
        /// frame. Logged so a run can be re-scored offline against a
        /// different `closedRatio` without re-collecting it.
        var earBaseline: Double = .nan
        var blinkHeld: Bool = false

        // Degrees. Raw is left NaN on a blink-held frame — there was no new
        // measurement, and writing the held value would fabricate one.
        var rawPitchDeg: Double = .nan
        var rawYawDeg: Double = .nan
        var filtPitchDeg: Double = .nan
        var filtYawDeg: Double = .nan

        // Absolute screen points; NaN before a calibration exists.
        var rawPredX: Double = .nan
        var rawPredY: Double = .nan
        var filtPredX: Double = .nan
        var filtPredY: Double = .nan
    }

    static let header: [String] = [
        "t_s", "timestamp_ms", "dropped_since",
        "latency_ms", "dispatch_ms",
        "pose_ms", "head_filter_ms", "eye_warp_ms", "face_warp_ms",
        "cnn_ms", "gaze_filter_ms", "publish_ms", "total_ms",
        "pose_ok", "face_crop_ok",
        "q_raw_x", "q_raw_y", "q_raw_z", "q_raw_w",
        "q_filt_x", "q_filt_y", "q_filt_z", "q_filt_w",
        "eye_raw_x_mm", "eye_raw_y_mm", "eye_raw_z_mm",
        "eye_filt_x_mm", "eye_filt_y_mm", "eye_filt_z_mm",
        "ear_left", "ear_right", "ear_mean", "ear_baseline", "blink_held",
        "raw_pitch_deg", "raw_yaw_deg", "filt_pitch_deg", "filt_yaw_deg",
        "raw_pred_x", "raw_pred_y", "filt_pred_x", "filt_pred_y",
    ]

    private let ioQueue = DispatchQueue(label: "gaze.instrumentation",
                                        qos: .utility)
    private let lock = NSLock()
    private var buffer: [Row] = []
    private var origin: CFTimeInterval?
    private let fileURL: URL
    private var wroteHeader = false
    /// Flush once the buffer reaches this many rows (~4 s at 60 Hz).
    private let flushThreshold = 240

    let enabled: Bool

    init(enabled: Bool = PipelineTuning.instrumentationEnabled) {
        self.enabled = enabled
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd_HHmmss"
        let dir = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("instrumentation", isDirectory: true)
        self.fileURL = dir.appendingPathComponent(
            "frames_\(fmt.string(from: Date())).csv")
        guard enabled else { return }
        try? FileManager.default.createDirectory(at: dir,
                                                 withIntermediateDirectories: true)
        print("[Instrumentation] \(fileURL.path)")
        print("[Instrumentation] \(PipelineTuning.describe())")
    }

    /// Append a row. Safe from any thread; never touches the disk inline.
    func record(_ row: Row) {
        guard enabled else { return }
        lock.lock()
        if origin == nil { origin = row.arrival }
        buffer.append(row)
        let ready = buffer.count >= flushThreshold ? takeAllLocked() : nil
        lock.unlock()
        if let ready = ready { write(ready) }
    }

    /// Force everything buffered to disk. Call when the camera stops.
    func flush() {
        guard enabled else { return }
        lock.lock()
        let ready = takeAllLocked()
        lock.unlock()
        if !ready.isEmpty { write(ready) }
    }

    // MARK: - Internals

    /// Caller must hold `lock`.
    private func takeAllLocked() -> [Row] {
        let out = buffer
        buffer.removeAll(keepingCapacity: true)
        return out
    }

    private func write(_ rows: [Row]) {
        let t0 = lock.withLockValue { origin } ?? rows.first?.arrival ?? 0
        ioQueue.async {
            var text = ""
            if !self.wroteHeader {
                text += Self.header.joined(separator: ",") + "\n"
                self.wroteHeader = true
            }
            for r in rows { text += Self.line(r, origin: t0) + "\n" }
            guard let data = text.data(using: .utf8) else { return }
            if let handle = try? FileHandle(forWritingTo: self.fileURL) {
                defer { try? handle.close() }
                _ = try? handle.seekToEnd()
                try? handle.write(contentsOf: data)
            } else {
                try? data.write(to: self.fileURL, options: .atomic)
            }
        }
    }

    private static func line(_ r: Row, origin: CFTimeInterval) -> String {
        let ms = { (s: Double) in f(s * 1000.0, 3) }
        let total = (r.publish > 0 && r.arrival > 0) ? (r.publish - r.arrival) : Double.nan
        var cells: [String] = [
            f(r.arrival - origin, 5),
            "\(r.timestampMs)",
            "\(r.droppedSince)",
            ms(total),
            ms(r.dtDispatch),
            ms(r.dtPose), ms(r.dtHeadFilter), ms(r.dtEyeWarp), ms(r.dtFaceWarp),
            ms(r.dtCNN), ms(r.dtGazeFilter), ms(r.dtPublish), ms(total),
            r.poseOK ? "1" : "0",
            r.faceCropOK ? "1" : "0",
        ]
        cells += quat(r.qRaw)
        cells += quat(r.qFiltered)
        cells += vec(r.eyeRaw)
        cells += vec(r.eyeFiltered)
        let earMean = EyeAspectRatio.Ratios(left: r.earLeft, right: r.earRight).mean
        cells += [f(r.earLeft, 5), f(r.earRight, 5), f(earMean, 5),
                  f(r.earBaseline, 5),
                  r.blinkHeld ? "1" : "0"]
        cells += [f(r.rawPitchDeg, 4), f(r.rawYawDeg, 4),
                  f(r.filtPitchDeg, 4), f(r.filtYawDeg, 4)]
        cells += [f(r.rawPredX, 2), f(r.rawPredY, 2),
                  f(r.filtPredX, 2), f(r.filtPredY, 2)]
        return cells.joined(separator: ",")
    }

    private static func quat(_ q: simd_quatd?) -> [String] {
        guard let q = q else { return ["", "", "", ""] }
        return [f(q.vector.x, 6), f(q.vector.y, 6),
                f(q.vector.z, 6), f(q.vector.w, 6)]
    }

    private static func vec(_ v: simd_double3?) -> [String] {
        guard let v = v else { return ["", "", ""] }
        return [f(v.x, 3), f(v.y, 3), f(v.z, 3)]
    }

    /// Non-finite values become empty cells, so pandas reads them as NaN
    /// rather than as a misleading zero.
    private static func f(_ x: Double, _ places: Int) -> String {
        guard x.isFinite else { return "" }
        return String(format: "%.\(places)f", x)
    }
}

private extension NSLock {
    func withLockValue<T>(_ body: () -> T) -> T {
        lock(); defer { unlock() }
        return body()
    }
}
