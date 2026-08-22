import Foundation
import simd

/// Head pose in the camera frame: `R, t` such that
/// `cameraPoint = R * facePoint + t` (translation in mm).
struct HeadPose {
    let rotation: simd_double3x3
    let translation: simd_double3

    /// Tait-Bryan Euler angles in **degrees**, derived from `rotation`.
    /// Conventions tuned empirically for this codebase's mirrored-front-camera setup:
    ///   - yaw: head turning left/right around the face's vertical Y axis
    ///   - pitch: head nodding up/down around the face's horizontal X axis
    ///   - roll: head tilting around the face's forward Z axis
    var euler: (yaw: Double, pitch: Double, roll: Double) {
        let R = rotation
        // R is column-major (simd convention): R[col][row].
        // For R = Ry(yaw) * Rx(pitch) * Rz(roll):
        //   R[0][2] = -sin(yaw)*cos(pitch)
        //   R[1][2] =  sin(pitch)
        //   R[2][2] =  cos(yaw)*cos(pitch)
        //   R[1][0] = -cos(pitch)*sin(roll)
        //   R[1][1] =  cos(pitch)*cos(roll)
        let r12 = R[1][2]
        let pitchRad = asin(max(-1.0, min(1.0, r12)))
        let yawRad   = atan2(-R[0][2], R[2][2])
        let rollRad  = atan2(-R[1][0], R[1][1])
        let toDeg = 180.0 / .pi
        return (yawRad * toDeg, pitchRad * toDeg, rollRad * toDeg)
    }

    static let identity = HeadPose(
        rotation: matrix_identity_double3x3,
        translation: simd_double3(0, 0, 600)
    )
}
