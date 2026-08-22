import Foundation
import simd
import MediaPipeTasksVision

/// Stage 2 of the gaze pipeline: from 478 face landmarks → head pose (R, t).
///
/// Picks 6 anchor landmarks (nose / chin / eye corners / mouth corners),
/// converts their normalized coordinates to pixels for the delivered
/// portrait+mirrored frame, and runs `SolvePnP.solve`.
///
/// Maintains the previous solution as a warm-start to make per-frame
/// refinement cheap and stable across small motions.
final class HeadPoseEstimator {
    private var lastPose: HeadPose = .identity
    /// Reason the most recent estimate failed (or nil on success). Lets the
    /// view model surface a debug message when Stage 2 silently breaks on
    /// device.
    private(set) var lastFailureReason: String?

    func estimate(landmarks: [NormalizedLandmark],
                  intrinsics: CameraIntrinsics) -> HeadPose? {
        guard landmarks.count >= CanonicalFaceModel.landmarkIndices.max()! + 1 else {
            lastFailureReason = "lm count \(landmarks.count)"
            return nil
        }

        var imagePoints: [CGPoint] = []
        imagePoints.reserveCapacity(CanonicalFaceModel.landmarkIndices.count)

        for idx in CanonicalFaceModel.landmarkIndices {
            let lm = landmarks[idx]
            // MediaPipe normalized [0,1] → pixel coordinates in delivered image.
            let px = CGFloat(lm.x) * intrinsics.imageWidth
            let py = CGFloat(lm.y) * intrinsics.imageHeight
            imagePoints.append(CGPoint(x: px, y: py))
        }

        let solution = SolvePnP.solve(
            modelPoints: CanonicalFaceModel.points3D,
            imagePoints: imagePoints,
            intrinsics: intrinsics,
            initial: lastPose
        )

        if let pose = solution {
            // Warm-start guard: if the pose looks broken (translation way off
            // or NaN), reset to identity for the next frame.
            let t = pose.translation
            let valid = t.z > 50 && t.z < 5000 && t.x.isFinite && t.y.isFinite
            lastPose = valid ? pose : .identity
            if !valid {
                lastFailureReason = String(format: "t=(%.0f,%.0f,%.0f)",
                                            t.x, t.y, t.z)
                return nil
            }
            lastFailureReason = nil
            return pose
        }
        // Reset the warm start if we failed to converge.
        lastPose = .identity
        lastFailureReason = SolvePnP.lastFailure ?? "solver nil"
        return nil
    }
}
