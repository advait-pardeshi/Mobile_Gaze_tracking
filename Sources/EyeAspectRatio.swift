import Foundation
import MediaPipeTasksVision

/// Eye-aspect-ratio (EAR) from the MediaPipe Face Mesh eyelid contour.
///
/// Soukupová & Čech (2016): for six points around one eye — two corners and
/// two upper/lower lid pairs —
///
///     EAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 · ‖p1 − p4‖)
///
/// The eye-corner distance in the denominator makes the ratio scale-free, so
/// one threshold holds whether the user is at 300 mm or 700 mm. A fully open
/// eye lands near 0.30 on this landmark set; a closed one near 0.10.
///
/// **Why this matters here.** A closed-eye crop is not a hard case for the
/// gaze CNN, it is an *out-of-distribution* one: ETH-XGaze contains almost no
/// closed eyes, so the network still emits a confident `(pitch, yaw)` with no
/// signal behind it. Those frames are the source of the large single-frame
/// excursions in the fixation logs. Detecting them upstream is cheaper and far
/// more reliable than trying to reject them downstream by magnitude.
enum EyeAspectRatio {

    struct Ratios {
        var left: Double
        var right: Double
        /// Mean of both eyes; falls back to whichever eye is finite.
        var mean: Double {
            switch (left.isFinite, right.isFinite) {
            case (true, true):  return (left + right) * 0.5
            case (true, false): return left
            case (false, true): return right
            case (false, false): return .nan
            }
        }
    }

    // Face Mesh eyelid contour indices.
    //   corners:  outer, inner
    //   lids:     (upper, lower) × 2, sampled either side of the iris.
    private static let leftCorners  = (outer: 33,  inner: 133)
    private static let leftLidsA    = (upper: 160, lower: 144)
    private static let leftLidsB    = (upper: 158, lower: 153)

    private static let rightCorners = (outer: 263, inner: 362)
    private static let rightLidsA   = (upper: 385, lower: 380)
    private static let rightLidsB   = (upper: 387, lower: 373)

    private static let maxIndex = 387

    /// `imageSize` is required, not optional: MediaPipe's normalized
    /// coordinates are divided by *different* extents in x and y, so on a
    /// 720×1280 portrait frame an unscaled EAR would be inflated by 16/9 and
    /// no fixed threshold would mean anything.
    static func compute(landmarks: [NormalizedLandmark],
                        imageSize: CGSize) -> Ratios {
        guard landmarks.count > maxIndex,
              imageSize.width > 0, imageSize.height > 0 else {
            return Ratios(left: .nan, right: .nan)
        }
        return Ratios(
            left: ear(landmarks, leftCorners, leftLidsA, leftLidsB, imageSize),
            right: ear(landmarks, rightCorners, rightLidsA, rightLidsB, imageSize)
        )
    }

    private static func ear(_ lm: [NormalizedLandmark],
                            _ corners: (outer: Int, inner: Int),
                            _ lidsA: (upper: Int, lower: Int),
                            _ lidsB: (upper: Int, lower: Int),
                            _ size: CGSize) -> Double {
        let width = distPx(lm[corners.outer], lm[corners.inner], size)
        guard width > 1e-6 else { return .nan }
        let a = distPx(lm[lidsA.upper], lm[lidsA.lower], size)
        let b = distPx(lm[lidsB.upper], lm[lidsB.lower], size)
        return (a + b) / (2.0 * width)
    }

    private static func distPx(_ a: NormalizedLandmark,
                               _ b: NormalizedLandmark,
                               _ size: CGSize) -> Double {
        let dx = Double(a.x - b.x) * Double(size.width)
        let dy = Double(a.y - b.y) * Double(size.height)
        return (dx * dx + dy * dy).squareRoot()
    }
}

/// Blink decision: a bounded gate against a per-user, adapting open-eye
/// baseline.
///
/// **What this has to separate.** A fixed EAR threshold cannot tell a blink
/// from a **downward gaze**. Looking at the bottom of the screen lowers the
/// upper lid over the iris and EAR falls to roughly 0.15–0.20 with the eye
/// wide open — the same range a threshold has to sit in to catch a real
/// closure. A ratio against the user's own baseline is better but not enough
/// on its own: 0.62 x a 0.30 baseline is 0.186, which is the fixed threshold
/// again under another name.
///
/// **What actually separates them is duration.** A blink is a fast transient,
/// 100–150 ms, three to five frames at 30 Hz. A downward gaze is a *state*
/// that lasts as long as the participant is looking there — the whole dwell,
/// seconds at a time. So the gate is bounded: it may suppress the CNN for a
/// blink's worth of frames and no longer. Past that the low EAR is taken as
/// the new normal, the baseline is pulled toward it, and the CNN runs again.
///
/// That bound is the part that matters. Without it the gate has no exit: while
/// it is closed the CNN never runs, so no new estimate arrives, so nothing can
/// re-open it. A participant looking at the bottom row would see the dot
/// freeze, then vanish, and stay gone until they looked back up.
///
/// `absoluteFloor` is the one unbounded case. Below it the eye really is shut
/// — no baseline ratio and no duration argument applies — so gating continues
/// for as long as it lasts.
struct BlinkDetector {

    /// Fraction of the running baseline below which the eye may be closed.
    static var closedRatio: Double { PipelineTuning.blinkEARRatio }
    /// Hard floor: below this the eye is shut regardless of baseline or
    /// duration. The only gate that never times out.
    static var absoluteFloor: Double { PipelineTuning.blinkEARFloor }
    /// Longest run of gated frames a blink is allowed to be. Past this the
    /// low EAR is a sustained lid position, not a blink.
    static var maxGatedFrames: Int { PipelineTuning.blinkMaxGatedFrames }
    /// Baseline used until a real one is learned. Overwritten by the first
    /// finite EAR, so this only covers the pre-roll.
    static let initialBaseline: Double = 0.28

    /// Smoothing when the current EAR is *above* the baseline (opening).
    private static let attack: Double = 0.10
    /// Smoothing when it is below but still open (a narrower eye).
    private static let decay: Double = 0.01
    /// Smoothing once the gate has timed out. Fast on purpose: the decision
    /// that this is a sustained lid position has already been made, and a
    /// slow adapt here would re-trip the gate on the very next frame.
    private static let reopenAdapt: Double = 0.25

    private var baseline: Double = BlinkDetector.initialBaseline
    private var learned = false
    private var gatedRun = 0

    /// Current open-eye baseline, for the instrumentation log.
    var currentBaseline: Double { baseline }

    mutating func reset() {
        baseline = BlinkDetector.initialBaseline
        learned = false
        gatedRun = 0
    }

    /// Feed this frame's mean EAR; returns true iff the CNN should be
    /// skipped. A non-finite EAR (no face, no landmarks) is *not* a blink —
    /// that frame has already failed upstream for other reasons.
    mutating func isBlink(meanEAR ear: Double) -> Bool {
        guard ear.isFinite else { gatedRun = 0; return false }
        if !learned {
            baseline = ear
            learned = true
        }

        // Genuinely shut. No timeout: holding here is correct for as long as
        // it lasts, and the pipeline drops the estimate separately once the
        // closure outlasts a plausible blink.
        if ear < Self.absoluteFloor {
            gatedRun += 1
            return true
        }

        if ear < Self.closedRatio * baseline {
            gatedRun += 1
            if gatedRun > Self.maxGatedFrames {
                // Too long to be a blink. Accept it as the working lid
                // position and let the CNN see the frame.
                baseline += (ear - baseline) * Self.reopenAdapt
                return false
            }
            return true
        }

        gatedRun = 0
        // Only open frames teach the baseline at the slow rates. Updating it
        // on gated frames would let a closure walk the threshold onto itself.
        baseline += (ear - baseline) * (ear > baseline ? Self.attack : Self.decay)
        return false
    }
}
