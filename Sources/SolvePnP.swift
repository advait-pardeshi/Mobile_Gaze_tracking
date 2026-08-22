import Foundation
import simd
import Accelerate

/// Iterative Gauss-Newton solver for the Perspective-n-Point problem.
///
/// Given N≥4 correspondences between 3D model points and 2D image points,
/// finds the rotation `R` and translation `t` that minimize reprojection
/// error under a pinhole projection with intrinsics K.
///
/// We parameterize rotation perturbations with axis-angle (Lie so(3)) so
/// the unknown vector is just 6-D: `(δθ_x, δθ_y, δθ_z, δt_x, δt_y, δt_z)`.
/// Each iteration:
///   1. Project current (R, t) → residual r ∈ R^{2N}
///   2. Build Jacobian J ∈ R^{2N×6}
///   3. Solve normal equations (J^T J) δ = -J^T r  with LAPACK dgesv
///   4. Update: R ← exp([δθ]_×) ⋅ R, t ← t + δt
///
/// 6 well-spread points (nose/chin/eyes/mouth) typically converge in ≤10
/// iterations. Returns nil if the solver diverges or the system is singular.
enum SolvePnP {
    /// Last failure cause for the most recent `solve(...)` call, or nil on
    /// success. Read by `HeadPoseEstimator` for diagnostics.
    static var lastFailure: String?

    static func solve(modelPoints: [simd_double3],
                      imagePoints: [CGPoint],
                      intrinsics: CameraIntrinsics,
                      initial: HeadPose = .identity,
                      maxIterations: Int = 30,
                      convergenceTolerance: Double = 1e-5) -> HeadPose? {
        lastFailure = nil

        precondition(modelPoints.count == imagePoints.count, "PnP: count mismatch")
        precondition(modelPoints.count >= 4, "PnP: need at least 4 points")

        var R = initial.rotation
        var t = initial.translation
        let n = modelPoints.count

        let fx = intrinsics.fx, fy = intrinsics.fy
        let cx = intrinsics.cx, cy = intrinsics.cy

        // Normal equations buffers (column-major for LAPACK).
        // A = J^T J  (6×6), b = -J^T r  (6×1).
        var A = [Double](repeating: 0, count: 36)
        var b = [Double](repeating: 0, count: 6)

        for _ in 0..<maxIterations {
            // Reset normal equations.
            A = [Double](repeating: 0, count: 36)
            b = [Double](repeating: 0, count: 6)

            for i in 0..<n {
                let P = modelPoints[i]
                let Pc = R * P + t
                let z = Pc.z
                if z <= 1.0 {
                    lastFailure = String(format: "z<=1 at pt %d (z=%.1f)", i, z)
                    return nil
                } // behind/at camera — bail

                let invZ = 1.0 / z
                let invZ2 = invZ * invZ

                // Residual = projected - observed
                let u = fx * Pc.x * invZ + cx
                let v = fy * Pc.y * invZ + cy
                let ru = u - Double(imagePoints[i].x)
                let rv = v - Double(imagePoints[i].y)

                // ∂(u, v)/∂Pc — 2×3
                let dudPc = (fx * invZ, 0.0,           -fx * Pc.x * invZ2)
                let dvdPc = (0.0,        fy * invZ,    -fy * Pc.y * invZ2)

                // ∂Pc/∂t = I (3×3), so ∂(u,v)/∂t = ∂(u,v)/∂Pc.
                // ∂Pc/∂θ for left-multiply update R' = exp([δθ]_×)⋅R:
                //   Pc' ≈ Pc + [δθ]_× · (Pc - t)  where (Pc - t) = R·P
                //   So the Jacobian ∂Pc/∂θ = -[(Pc - t)]_×
                //   NOT -[Pc]_× — using Pc instead of Pc-t was a bug that
                //   caused 600mm of error in the z-gradient, making the
                //   solver diverge to ±tens-of-metres.
                let q = Pc - t   // = R * modelPoint (rotated, un-translated)
                let cross = (
                    (0.0,   q.z,  -q.y),
                    (-q.z,  0.0,   q.x),
                    (q.y,  -q.x,   0.0)
                )

                // Build per-point 2×6 Jacobian rows: [∂(u,v)/∂θ | ∂(u,v)/∂t]
                // ∂(u,v)/∂θ = ∂(u,v)/∂Pc ⋅ (-[Pc]_×)
                var Ju = [Double](repeating: 0, count: 6)
                var Jv = [Double](repeating: 0, count: 6)

                // First 3 columns: rotation part.
                for k in 0..<3 {
                    let crossCol: (Double, Double, Double)
                    switch k {
                    case 0: crossCol = (cross.0.0, cross.1.0, cross.2.0)
                    case 1: crossCol = (cross.0.1, cross.1.1, cross.2.1)
                    default: crossCol = (cross.0.2, cross.1.2, cross.2.2)
                    }
                    Ju[k] = dudPc.0 * crossCol.0 + dudPc.1 * crossCol.1 + dudPc.2 * crossCol.2
                    Jv[k] = dvdPc.0 * crossCol.0 + dvdPc.1 * crossCol.1 + dvdPc.2 * crossCol.2
                }
                // Last 3 columns: translation part.
                Ju[3] = dudPc.0; Ju[4] = dudPc.1; Ju[5] = dudPc.2
                Jv[3] = dvdPc.0; Jv[4] = dvdPc.1; Jv[5] = dvdPc.2

                // Accumulate A += J^T J, b += -J^T r.
                for r in 0..<6 {
                    for c in 0..<6 {
                        // Column-major: A[c*6 + r]
                        A[c * 6 + r] += Ju[r] * Ju[c] + Jv[r] * Jv[c]
                    }
                    b[r] += -(Ju[r] * ru + Jv[r] * rv)
                }
            }

            // Solve A δ = b (column-major).
            var nLA: __CLPK_integer = 6
            var nrhs: __CLPK_integer = 1
            var lda: __CLPK_integer = 6
            var ldb: __CLPK_integer = 6
            var info: __CLPK_integer = 0
            var ipiv = [__CLPK_integer](repeating: 0, count: 6)

            // Add a tiny Levenberg-Marquardt-style damping to A's diagonal for
            // numerical stability when the camera-Z gradient is degenerate.
            for d in 0..<6 { A[d * 6 + d] += 1e-9 }

            var Acopy = A
            var bcopy = b
            dgesv_(&nLA, &nrhs, &Acopy, &lda, &ipiv, &bcopy, &ldb, &info)
            if info != 0 {
                lastFailure = "dgesv info=\(info)"
                return nil
            }

            let dTheta = simd_double3(bcopy[0], bcopy[1], bcopy[2])
            let dT     = simd_double3(bcopy[3], bcopy[4], bcopy[5])

            // Apply update.
            R = expSO3(dTheta) * R
            t = t + dT

            if simd_length(dTheta) + simd_length(dT) < convergenceTolerance { break }
        }

        return HeadPose(rotation: R, translation: t)
    }

    /// Rodrigues' formula: axis-angle vector ω → rotation matrix.
    private static func expSO3(_ omega: simd_double3) -> simd_double3x3 {
        let theta = simd_length(omega)
        if theta < 1e-12 { return matrix_identity_double3x3 }
        let k = omega / theta
        // Skew-symmetric [k]_×
        let K = simd_double3x3(columns: (
            simd_double3( 0.0,  k.z, -k.y),
            simd_double3(-k.z, 0.0,  k.x),
            simd_double3( k.y, -k.x, 0.0)
        ))
        let s = sin(theta)
        let c = cos(theta)
        // I + sin(θ)·K + (1-cos(θ))·K²
        return matrix_identity_double3x3 + s * K + (1 - c) * (K * K)
    }
}
