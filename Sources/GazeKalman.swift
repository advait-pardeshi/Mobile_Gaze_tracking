import Foundation
import CoreGraphics
import simd

/// Stage 6 — Kalman smoothing of the post-calibration screen point.
///
/// Constant-velocity model in **UIKit point space**:
///   state x = [px, py, vx, vy]ᵀ   (points, points/second)
///   F(dt) = ⎡1 0 dt  0⎤            H = ⎡1 0 0 0⎤
///           ⎢0 1  0 dt⎥                 ⎣0 1 0 0⎦
///           ⎢0 0  1  0⎥
///           ⎣0 0  0  1⎦
///
/// Tuning lives in `Config`. Larger `processNoise*` → more responsive but
/// jittery; larger `measurementNoise` → smoother but laggier.
final class GazeKalman {

    struct Config {
        /// σ² for x/y process noise (points²). How much per-frame drift the
        /// model expects in the position component.
        var positionProcessNoise: Float = 4.0
        /// σ² for vx/vy process noise (points²/s²). How quickly velocity can
        /// change frame-to-frame.
        var velocityProcessNoise: Float = 1500.0
        /// σ² for the position measurement (points²). Set this near the
        /// observed standard-deviation² of the raw post-calibration point.
        var measurementNoise:     Float = 900.0

        static let `default` = Config()
    }

    // MARK: - Public

    init(config: Config = .default) {
        self.config = config
        rebuildNoise()
    }

    var config: Config {
        didSet { rebuildNoise() }
    }

    private(set) var isInitialized = false

    /// Reset the filter — call on face loss or when calibration changes.
    func reset() {
        state = .zero
        P = matrix_identity_float4x4
        isInitialized = false
    }

    /// Feed a raw point, get back the smoothed estimate.
    /// - Parameter dt: seconds since the previous `update` call. Clamped
    ///   internally to avoid blow-up on first frame after a long pause.
    func update(measurement z: CGPoint, dt: TimeInterval) -> CGPoint {
        let zf = simd_float2(Float(z.x), Float(z.y))

        if !isInitialized {
            state = simd_float4(zf.x, zf.y, 0, 0)
            // Loose initial covariance so the first few corrections move fast.
            P = simd_float4x4(diagonal: simd_float4(
                config.measurementNoise,
                config.measurementNoise,
                config.velocityProcessNoise,
                config.velocityProcessNoise))
            isInitialized = true
            return z
        }

        let dtf = Float(max(1.0 / 240.0, min(0.2, dt)))
        predict(dt: dtf)
        correct(z: zf)
        return CGPoint(x: CGFloat(state.x), y: CGFloat(state.y))
    }

    // MARK: - Internals

    private var state = simd_float4.zero        // [px, py, vx, vy]
    private var P     = matrix_identity_float4x4 // state covariance
    private var Q     = matrix_identity_float4x4 // process noise
    private var R     = matrix_identity_float2x2 // measurement noise

    /// H = [[1,0,0,0],[0,1,0,0]]; stored implicitly via index access.
    private func predict(dt: Float) {
        // x = F·x
        state.x += state.z * dt
        state.y += state.w * dt
        // P = F·P·Fᵀ + Q  (constant-velocity F closed form)
        //  Let P = [[A B],[Bᵀ C]] with 2×2 blocks; then
        //    P' = [[A + dt(B+Bᵀ) + dt²C,  B + dt·C ],
        //          [Bᵀ + dt·C,             C        ]] + Q
        let p = P
        let dt2 = dt * dt

        // Extract 2×2 blocks (column-major: P.columns.i[j] = P[j,i]).
        // A = top-left, B = top-right, C = bottom-right.
        let A00 = p[0,0], A01 = p[1,0], A10 = p[0,1], A11 = p[1,1]
        let B00 = p[2,0], B01 = p[3,0], B10 = p[2,1], B11 = p[3,1]
        let Bt00 = p[0,2], Bt01 = p[1,2], Bt10 = p[0,3], Bt11 = p[1,3]
        let C00 = p[2,2], C01 = p[3,2], C10 = p[2,3], C11 = p[3,3]

        let newA00 = A00 + dt * (B00 + Bt00) + dt2 * C00
        let newA01 = A01 + dt * (B01 + Bt01) + dt2 * C01
        let newA10 = A10 + dt * (B10 + Bt10) + dt2 * C10
        let newA11 = A11 + dt * (B11 + Bt11) + dt2 * C11

        let newB00 = B00 + dt * C00
        let newB01 = B01 + dt * C01
        let newB10 = B10 + dt * C10
        let newB11 = B11 + dt * C11

        let newBt00 = Bt00 + dt * C00
        let newBt01 = Bt01 + dt * C01
        let newBt10 = Bt10 + dt * C10
        let newBt11 = Bt11 + dt * C11

        // Reassemble + add Q.
        P = simd_float4x4(columns: (
            simd_float4(newA00,  newA10,  newBt00, newBt10),
            simd_float4(newA01,  newA11,  newBt01, newBt11),
            simd_float4(newB00,  newB10,  C00,     C10),
            simd_float4(newB01,  newB11,  C01,     C11)
        )) + Q
    }

    /// Standard Kalman update with H = [I₂ | 0₂].
    private func correct(z: simd_float2) {
        let p = P
        // S = H·P·Hᵀ + R = top-left 2×2 of P + R
        let S = simd_float2x2(columns: (
            simd_float2(p[0,0], p[0,1]),
            simd_float2(p[1,0], p[1,1])
        )) + R

        let det = S[0,0] * S[1,1] - S[0,1] * S[1,0]
        guard abs(det) > 1e-9 else { return }
        let invDet = 1 / det
        let Sinv = simd_float2x2(columns: (
            simd_float2( S[1,1], -S[0,1]) * invDet,
            simd_float2(-S[1,0],  S[0,0]) * invDet
        ))

        // K = P·Hᵀ·S⁻¹  — only the first two columns of P matter (Hᵀ picks them).
        // PHt is 4×2 = columns 0 and 1 of P.
        let PHt_c0 = simd_float4(p[0,0], p[0,1], p[0,2], p[0,3])
        let PHt_c1 = simd_float4(p[1,0], p[1,1], p[1,2], p[1,3])

        // K columns = PHt · Sinv (4×2 = 4×2 · 2×2).
        let K_c0 = PHt_c0 * Sinv[0,0] + PHt_c1 * Sinv[0,1]
        let K_c1 = PHt_c0 * Sinv[1,0] + PHt_c1 * Sinv[1,1]

        // Innovation y = z − H·x.
        let innov = simd_float2(z.x - state.x, z.y - state.y)

        // x = x + K·y
        state = state + K_c0 * innov.x + K_c1 * innov.y

        // P = (I − K·H)·P. K·H is 4×4 with non-zero only in its first two columns,
        // so for column j of P the update is: col_j − (K_c0·P[0,j] + K_c1·P[1,j]).
        P = simd_float4x4(columns: (
            p.columns.0 - (K_c0 * p[0,0] + K_c1 * p[0,1]),
            p.columns.1 - (K_c0 * p[1,0] + K_c1 * p[1,1]),
            p.columns.2 - (K_c0 * p[2,0] + K_c1 * p[2,1]),
            p.columns.3 - (K_c0 * p[3,0] + K_c1 * p[3,1])
        ))
    }

    private func rebuildNoise() {
        let qp = config.positionProcessNoise
        let qv = config.velocityProcessNoise
        Q = simd_float4x4(diagonal: simd_float4(qp, qp, qv, qv))

        let rm = config.measurementNoise
        R = simd_float2x2(diagonal: simd_float2(rm, rm))
    }
}
