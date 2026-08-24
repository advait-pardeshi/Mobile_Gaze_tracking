import Foundation

/// Central switchboard for the `perf-and-stability` branch.
///
/// Everything here is a compile-time-ish constant rather than a settings
/// screen on purpose: these knobs change what the *logged data* means, so a
/// run has to be able to state which configuration produced it. `describe()`
/// is written into every instrumentation file and every experiment bundle.
enum PipelineTuning {

    /// Which post-CNN smoother drives the rendered gaze dot.
    ///
    /// Both paths log identical raw / filtered streams, so an A/B is a
    /// matter of flipping this and re-running Experiment 2.
    enum Smoother: String {
        /// One Euro on `(pitch, yaw)` + One Euro on the eye-position vector,
        /// applied **before** projection. Default on this branch.
        case oneEuro
        /// The previous Stage 6 constant-velocity Kalman on the projected
        /// screen point. Kept for A/B only.
        case kalman
    }

    static let smoother: Smoother = .oneEuro

    /// Which stream Experiment 2 (fixation stability) actually scores.
    ///
    /// `.raw` is the estimator's noise floor: the per-frame CNN output, no
    /// temporal smoothing, no blink hold. `.filtered` is the stream the
    /// on-screen cursor is drawn from — post One Euro (or Kalman), blink
    /// holds included.
    ///
    /// These measure different things. `.filtered` will always report better
    /// dispersion and RMS than `.raw`, because shrinking frame-to-frame
    /// variance is precisely what the smoother does; the number then
    /// describes the filter as much as the tracker, and turning the cutoff
    /// down would "improve" it further without the tracker changing at all.
    /// Use `.filtered` when the question is *what the user experiences*
    /// (the cursor is what they see and act on) and `.raw` when the question
    /// is how good the estimator is. Whichever is chosen is written into
    /// every run's `prediction_source`, so no run is ambiguous.
    enum ScoredStream: String {
        case raw
        case filtered
    }

    static let fixationScoredStream: ScoredStream = .filtered

    // MARK: - One Euro parameters

    /// Gaze angle filter. `minCutoff` sets the floor smoothing during a
    /// fixation; `beta` sets how fast it lets go during a saccade. Tuned so a
    /// held fixation is quiet without visibly lagging a deliberate look
    /// across the screen.
    static let gazeAngleMinCutoff: Double = 1.2      // Hz
    static let gazeAngleBeta:      Double = 0.35
    static let gazeAngleDCutoff:   Double = 1.0      // Hz

    /// Eye-position (PnP translation) filter. Much lower cutoff: the head
    /// physically cannot move as fast as the gaze angle can, so most of what
    /// this sees is PnP depth noise.
    static let eyePositionMinCutoff: Double = 0.5    // Hz
    static let eyePositionBeta:      Double = 0.02
    static let eyePositionDCutoff:   Double = 1.0    // Hz

    /// Head-rotation quaternion filter, applied *before* the normalization
    /// crop so the 224×224 face image stops shimmering frame-to-frame.
    static let headPoseMinCutoff: Double = 0.8       // Hz
    static let headPoseBeta:      Double = 0.10
    static let headPoseDCutoff:   Double = 1.0       // Hz

    // MARK: - Blink gating

    /// Eye-aspect-ratio below which the frame is treated as a blink and the
    /// CNN is skipped entirely. A fully-open eye sits near 0.30 on this
    /// landmark set; a closed one near 0.10.
    static let blinkEARThreshold: Double = 0.18

    /// Longest run of consecutive blink-gated frames that will keep holding
    /// the previous estimate. Past this the estimate is dropped rather than
    /// frozen — a closed eye for a full second is not a blink, and holding
    /// through it would paint a confidently wrong dot.
    static let blinkMaxHeldFrames: Int = 12

    // MARK: - Debug throttling

    /// Publish the debug crops (eye strip, normalized face) on every Nth
    /// processed frame. The eye strip is *also skipped entirely* on the
    /// other frames — it feeds no downstream stage, so warping it just to
    /// throw it away was pure cost.
    static let debugCropPublishInterval: Int = 5

    // MARK: - Instrumentation

    /// Write the per-frame stage-timing CSV to
    /// `Documents/instrumentation/`. Temporary — flip to `false` once the
    /// time budget question is settled.
    static let instrumentationEnabled = true

    static func describe() -> String {
        "smoother=\(smoother.rawValue) "
        + "exp2Scored=\(fixationScoredStream.rawValue) "
        + "gaze(minCutoff=\(gazeAngleMinCutoff),beta=\(gazeAngleBeta)) "
        + "eye(minCutoff=\(eyePositionMinCutoff),beta=\(eyePositionBeta)) "
        + "head(minCutoff=\(headPoseMinCutoff),beta=\(headPoseBeta)) "
        + "blinkEAR=\(blinkEARThreshold) "
        + "debugEvery=\(debugCropPublishInterval)"
    }
}

/// Minimal test-and-set flag usable from any thread.
///
/// Deployment target is iOS 15, so `OSAllocatedUnfairLock` is unavailable;
/// `NSLock` around a `Bool` is more than fast enough for one acquire per
/// camera frame.
final class AtomicFlag {
    private let lock = NSLock()
    private var raised = false

    /// Raises the flag and returns `true` iff it was previously lowered.
    func tryAcquire() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if raised { return false }
        raised = true
        return true
    }

    func release() {
        lock.lock()
        raised = false
        lock.unlock()
    }
}

/// Thread-safe counter for dropped frames, so the instrumentation can report
/// how much backpressure actually fired.
final class AtomicCounter {
    private let lock = NSLock()
    private var value = 0

    func increment() {
        lock.lock(); value += 1; lock.unlock()
    }

    /// Read and zero in one step.
    func drain() -> Int {
        lock.lock()
        defer { value = 0; lock.unlock() }
        return value
    }
}
