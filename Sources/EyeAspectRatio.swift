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

/// Blink decision with a per-user, slowly-adapting open-eye baseline.
///
/// A fixed EAR threshold cannot separate a blink from a **downward gaze**.
/// Looking at the bottom of the screen lowers the upper lid over the iris,
/// and EAR falls to roughly 0.15–0.20 — below any threshold set low enough
/// to catch a real closure on some users, and squarely on top of 0.18. The
/// pipeline then gates the CNN off for exactly the frames the participant is
/// trying to reach the bottom row with, holds the previous estimate, and the
/// dot never gets there.
///
/// What actually distinguishes the two is *depth relative to that person's
/// own open eye*: a lid lowered for a downward gaze keeps ~65–80 % of the
/// open aperture, a blink collapses to ~30 %. So the gate is a fraction of a
/// running baseline, with an absolute floor as a backstop for the first few
/// frames and for unusually narrow eyes.
///
/// The baseline attacks fast and decays slowly: it should follow a user
/// settling into position within a second, but must not be dragged down by a
/// blink (or by a long downward dwell) until it starts calling blinks open.
struct BlinkDetector {

    /// Fraction of the running baseline below which the eye counts as closed.
    static var closedRatio: Double { PipelineTuning.blinkEARRatio }
    /// Absolute EAR below which the eye is closed regardless of baseline —
    /// catches the case where the baseline itself was learned on a blink.
    static var absoluteFloor: Double { PipelineTuning.blinkEARFloor }
    /// Baseline used until a real one is learned. Overwritten by the first
    /// finite EAR, so this only covers the pre-roll.
    static let initialBaseline: Double = 0.28

    /// Smoothing when the current EAR is *above* the baseline (opening).
    private static let attack: Double = 0.10
    /// Smoothing when it is *below* (closing). ~1/10th the attack, so a
    /// blink barely moves the baseline but a genuinely narrower eye still
    /// converges over a few seconds.
    private static let decay: Double = 0.01

    private var baseline: Double = BlinkDetector.initialBaseline
    private var learned = false

    /// Current open-eye baseline, for the instrumentation log.
    var currentBaseline: Double { baseline }

    mutating func reset() {
        baseline = BlinkDetector.initialBaseline
        learned = false
    }

    /// Feed this frame's mean EAR; returns true iff it should be gated as a
    /// blink. A non-finite EAR (no face, no landmarks) is *not* a blink —
    /// that frame has already failed upstream for other reasons.
    mutating func isBlink(meanEAR ear: Double) -> Bool {
        guard ear.isFinite else { return false }
        if !learned {
            baseline = ear
            learned = true
        }
        let threshold = max(Self.absoluteFloor, Self.closedRatio * baseline)
        let closed = ear < threshold
        // Only open frames teach the baseline. Updating it on closed frames
        // too would let a long closure walk the threshold down onto itself.
        if !closed {
            let a = ear > baseline ? Self.attack : Self.decay
            baseline += (ear - baseline) * a
        }
        return closed
    }
}
