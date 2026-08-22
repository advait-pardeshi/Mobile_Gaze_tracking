import Foundation
import MediaPipeTasksVision

/// Iris-based pupil-size proxy from MediaPipe's refined iris landmarks.
///
/// MediaPipe Face Mesh with iris refinement emits 478 landmarks; indices
/// 468–472 are the left-iris ring (468 = center, 469/470/471/472 =
/// right/top/left/bottom) and 473–477 are the right-iris ring. MediaPipe
/// does not segment the pupil itself, so the iris diameter is the closest
/// available proxy — reported in image pixels so it's comparable across
/// frames (the raw landmarks are normalized [0,1] to the image).
enum PupilMeasure {

    /// Left- and right-eye iris diameters in image pixels, or NaN when the
    /// landmark set is too short (iris refinement off / no face).
    struct Diameters {
        var leftPx: Double
        var rightPx: Double
        /// Mean of the two eyes; NaN if either is missing.
        var meanPx: Double {
            guard leftPx.isFinite, rightPx.isFinite else { return .nan }
            return (leftPx + rightPx) * 0.5
        }
    }

    static func diameters(landmarks: [NormalizedLandmark],
                          imageSize: CGSize) -> Diameters {
        guard landmarks.count >= 478,
              imageSize.width > 0, imageSize.height > 0 else {
            return Diameters(leftPx: .nan, rightPx: .nan)
        }
        let left  = ringDiameterPx(landmarks, 469, 470, 471, 472, imageSize)
        let right = ringDiameterPx(landmarks, 474, 475, 476, 477, imageSize)
        return Diameters(leftPx: left, rightPx: right)
    }

    /// Mean of horizontal (right↔left) and vertical (top↔bottom) ring
    /// spans, in pixels.
    private static func ringDiameterPx(_ lm: [NormalizedLandmark],
                                       _ rIdx: Int, _ tIdx: Int,
                                       _ lIdx: Int, _ bIdx: Int,
                                       _ size: CGSize) -> Double {
        let h = distPx(lm[rIdx], lm[lIdx], size)
        let v = distPx(lm[tIdx], lm[bIdx], size)
        return (h + v) * 0.5
    }

    private static func distPx(_ a: NormalizedLandmark,
                               _ b: NormalizedLandmark,
                               _ size: CGSize) -> Double {
        let dx = Double(a.x - b.x) * Double(size.width)
        let dy = Double(a.y - b.y) * Double(size.height)
        return (dx * dx + dy * dy).squareRoot()
    }
}
