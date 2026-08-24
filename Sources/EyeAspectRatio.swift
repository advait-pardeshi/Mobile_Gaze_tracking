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
