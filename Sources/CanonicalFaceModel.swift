import Foundation
import simd

/// 3D positions (mm) of 6 landmarks on an "average" face, used as the
/// known model for `solvePnP`. Origin at nose tip. Units are millimeters.
///
/// The X coordinate is **flipped** versus the textbook anatomical convention
/// (where face's anatomical right is +X) to compensate for our front-camera
/// horizontal mirroring. With this flipped model + standard projection
/// equations, the canonical pose (face directly facing the mirrored camera)
/// is approximately R = identity, t ≈ (0, 0, 600), which makes the
/// Gauss-Newton refiner converge from R = I.
enum CanonicalFaceModel {
    /// MediaPipe Face Mesh landmark indices, in the same order as `points3D`.
    static let landmarkIndices: [Int] = [
        1,    // nose tip
        152,  // chin
        33,   // anatomical-left eye outer corner   (appears LEFT in mirrored image)
        263,  // anatomical-right eye outer corner  (appears RIGHT in mirrored image)
        61,   // anatomical-left mouth corner       (appears LEFT in mirrored image)
        291,  // anatomical-right mouth corner      (appears RIGHT in mirrored image)
    ]

    /// 3D points in face-local coordinates (mm).
    /// Y+ = up, Z+ = out of face (toward viewer). X+ flipped — see file header.
    static let points3D: [simd_double3] = [
        simd_double3(  0.0,    0.0,    0.0),   // 1   nose tip
        simd_double3(  0.0,  -63.0,  -12.5),   // 152 chin
        simd_double3( -43.0,  32.0,  -26.0),   // 33  → left side of mirrored image
        simd_double3(  43.0,  32.0,  -26.0),   // 263 → right side of mirrored image
        simd_double3( -28.0, -28.0,  -24.0),   // 61
        simd_double3(  28.0, -28.0,  -24.0),   // 291
    ]

    /// 3D axis tips (mm) used to render the head-pose visualization.
    /// 60mm so the axes are clearly visible projecting from the nose.
    static let axisLength: Double = 60.0
    static let axisTips: [simd_double3] = [
        simd_double3(0, 0, 0),                // origin
        simd_double3(axisLength, 0, 0),       // X (red) — face's anatomical-left dir
        simd_double3(0, axisLength, 0),       // Y (green) — up
        simd_double3(0, 0, axisLength),       // Z (blue) — out of face (toward camera)
    ]

    /// 3D positions of the left and right eye centers in face-local coords (mm).
    /// Stage 3 (eye normalization) constructs a separate virtual camera per eye
    /// that "looks straight at" these points, so the eye lands at the principal
    /// point of the normalized image regardless of head pose.
    ///
    /// These are roughly the midpoint between each eye's outer corner (33/263)
    /// and inner corner (133/362) on an average face, with X flipped to match
    /// the rest of `points3D`.
    ///
    /// `leftEyeCenter3D` corresponds to the eye on the LEFT of the mirrored
    /// preview image (anatomical-right eye in the user's body frame).
    static let leftEyeCenter3D  = simd_double3(-30.0, 32.0, -22.0)
    static let rightEyeCenter3D = simd_double3( 30.0, 32.0, -22.0)

    /// 3D face-center anchor used by Stage 3 face normalization (Phase 4
    /// ETH-XGaze path). Matches the plgaze definition for ETH-XGaze: the
    /// average of the four eye corners and the two nose alar points (mp
    /// landmarks 33, 133, 362, 263, 240, 460). Computed against the
    /// X-flipped face model so it stays in the same coordinate frame as
    /// the rest of this file.
    static let faceCenter3D = simd_double3(0.0, 26.0, -19.0)
}
