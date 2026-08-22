import Foundation
import simd
import CoreGraphics

/// Stage 5 geometry: project a unit gaze ray (camera frame) onto the phone
/// screen, and fit the screen pose from calibration samples.
///
/// PIPELINE.md §Stage 5 specifies:
/// ```
///   Sg = SRotG · (μ · g) + t          μ = -tz / gz
/// ```
/// where `g` is a unit gaze vector in camera coords, `t = (tx, ty, tz)` is
/// the screen center's pose in camera coords, and `SRotG = diag(-1, -1, 1)`
/// flips the camera's mirrored x/y to match the screen's UIKit-style frame.
///
/// Substituting μ into the x/y equations collapses to:
/// ```
///   Sg.x = tx + tz · (gx / gz)
///   Sg.y = ty + tz · (gy / gz)
/// ```
/// — **linear** in `t`. We fit by stacking two rows per sample into the
/// normal equations `(AᵀA) · t = Aᵀb` (3×3, solved by direct inverse).
///
/// We don't include the eye position in the projection; following PIPELINE.md
/// literally, the gaze emanates from the camera origin and the per-target
/// local offsets compensate for the resulting head-distance bias.
///
/// **Unit choice.** This file works in **screen points** (UIKit logical
/// units), not millimetres — sidesteps per-device PPI lookups. The fit is
/// empirical, so any consistent unit works; only the geometric interpretation
/// of `tz` (which we don't use downstream) is lost.
enum ScreenMapper {

    /// Camera-x and -y point opposite the screen's after the front-camera mirror.
    /// Diagonal, so `SRotG^T = SRotG` and `SRotG^2 = I`.
    static let SRotG = simd_double3x3(diagonal: simd_double3(-1, -1, 1))

    /// Project a unit gaze vector to a screen-center-relative 2D point in
    /// the same units as the calibration targets used to fit `t`. Returns
    /// `(NaN, NaN)` for gazes nearly parallel to the screen plane (gz ≈ 0).
    static func project(gazeCam g: simd_double3,
                        translation t: simd_double3) -> simd_double2 {
        guard abs(g.z) > 1e-3 else {
            return simd_double2(.nan, .nan)
        }
        return simd_double2(
            t.x + t.z * g.x / g.z,
            t.y + t.z * g.y / g.z
        )
    }

    /// Linear least-squares fit of `t = (tx, ty, tz)` from calibration data.
    /// Returns nil if the system is rank-deficient (insufficient gaze
    /// diversity — e.g., user fixated only one or two dots).
    static func fit(samples: [(gaze: simd_double3,
                                target: simd_double2)]) -> simd_double3? {
        // simd_double3x3 zero matrix; we accumulate (AᵀA) and (Aᵀb).
        var ata = simd_double3x3()
        var atb = simd_double3()
        var n = 0
        for s in samples {
            // Reject samples where the ray is nearly parallel to the screen.
            guard abs(s.gaze.z) > 0.1 else { continue }
            let r = s.gaze.x / s.gaze.z
            let q = s.gaze.y / s.gaze.z
            let rowX = simd_double3(1, 0, r)   // contributes to sx equation
            let rowY = simd_double3(0, 1, q)   // contributes to sy equation
            // Outer-product update: AᵀA += rowᵀrow (per row), columnwise.
            ata.columns.0 += rowX * rowX.x + rowY * rowY.x
            ata.columns.1 += rowX * rowX.y + rowY * rowY.y
            ata.columns.2 += rowX * rowX.z + rowY * rowY.z
            atb += rowX * s.target.x + rowY * s.target.y
            n += 1
        }
        guard n >= 4 else { return nil }
        let det = ata.determinant
        guard abs(det) > 1e-9 else { return nil }
        return ata.inverse * atb
    }

    /// 3×3 grid of calibration targets in screen-center-relative points.
    /// Inset to 80% of the screen extent so the dots stay clear of the
    /// safe-area edges.
    static func standardTargets(screenSize: CGSize) -> [simd_double2] {
        let halfW = Double(screenSize.width)  * 0.40
        let halfH = Double(screenSize.height) * 0.40
        var pts: [simd_double2] = []
        pts.reserveCapacity(9)
        for r in [-1.0, 0.0, 1.0] {        // top, middle, bottom
            for c in [-1.0, 0.0, 1.0] {    // left, centre, right
                pts.append(simd_double2(c * halfW, r * halfH))
            }
        }
        return pts
    }
}
