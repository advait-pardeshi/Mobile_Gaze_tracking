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


    // MARK: - Robust fit

    /// Outcome of a robust fit, with the diagnostics needed to decide whether
    /// the fit is trustworthy *before* an experiment consumes it.
    struct FitQuality {
        let translation: simd_double3
        /// Dots that survived outlier rejection and drove the final fit.
        let usedDotIndices: [Int]
        let totalDots: Int
        /// Residual `|predicted - target|` per used dot, screen points.
        let residualPoints: [Double]
        /// Dots rejected as outliers, by original index.
        let rejectedDotIndices: [Int]

        var medianResidualPoints: Double {
            ScreenMapper.median(residualPoints) ?? .nan
        }
        var maxResidualPoints: Double { residualPoints.max() ?? .nan }
    }

    /// Robust replacement for `fit(samples:)`.
    ///
    /// `fit(samples:)` is an unweighted least-squares over every retained
    /// sample, which has two failure modes this addresses:
    ///
    ///   1. **Per-dot weighting is accidental.** A dot that happened to yield
    ///      more valid frames pulls `t` harder than one that yielded fewer,
    ///      even though the 9-dot design intends them to count equally. We
    ///      collapse each dot to the component-wise **median** of its gaze
    ///      ratios `(gx/gz, gy/gz)` first — which also discards blinks and
    ///      the saccade into/out of the dot for free.
    ///   2. **One bad dot corrupts the whole session.** If the participant
    ///      wasn't actually fixating one dot, least-squares spreads that error
    ///      across `t` and biases *every* prediction on screen by a constant
    ///      offset — which is exactly the flat-across-eccentricity,
    ///      shifted-between-runs signature seen in the grid-experiment logs.
    ///      We fit, score each dot's residual, drop outliers by MAD, and refit.
    ///
    /// Returns nil when too few dots survive to trust the result.
    static func fitRobust(perDot: [(gazes: [simd_double3],
                                     target: simd_double2)]) -> FitQuality? {
        // Stage 1: collapse each dot to a robust ratio pair.
        var rows: [(r: Double, q: Double, target: simd_double2, dot: Int)] = []
        for (i, d) in perDot.enumerated() {
            var rs: [Double] = [], qs: [Double] = []
            for g in d.gazes where abs(g.z) > 0.1 {
                rs.append(g.x / g.z)
                qs.append(g.y / g.z)
            }
            guard let r = median(rs), let q = median(qs) else { continue }
            rows.append((r, q, d.target, i))
        }
        guard rows.count >= 4 else { return nil }

        // Stage 2: first pass over all surviving dots.
        guard let t0 = solveT(rows: rows) else { return nil }

        // Stage 3: per-dot residual, then MAD-based rejection.
        let res0 = rows.map { row -> Double in
            simd_distance(simd_double2(t0.x + t0.z * row.r,
                                       t0.y + t0.z * row.q), row.target)
        }
        let med = median(res0) ?? 0
        let mad = median(res0.map { abs($0 - med) }) ?? 0
        // 1.4826·MAD ≈ σ for normal data; 3σ is the usual outlier line.
        // The 20 pt floor stops a very tight fit from rejecting dots that are
        // fine in absolute terms — we want to catch broken dots, not trim
        // healthy scatter.
        let threshold = max(med + 3.0 * 1.4826 * mad, med + 20.0)

        var kept: [(r: Double, q: Double, target: simd_double2, dot: Int)] = []
        var rejected: [Int] = []
        for (row, r) in zip(rows, res0) {
            if r <= threshold { kept.append(row) } else { rejected.append(row.dot) }
        }

        // Stage 4: refit on survivors. Below 6 of 9 dots the fit isn't
        // sampled broadly enough to be worth handing to an experiment —
        // fail loudly instead of returning a plausible-looking `t`.
        guard kept.count >= 6, let t = solveT(rows: kept) else { return nil }

        let resFinal = kept.map { row -> Double in
            simd_distance(simd_double2(t.x + t.z * row.r,
                                       t.y + t.z * row.q), row.target)
        }
        return FitQuality(translation: t,
                          usedDotIndices: kept.map(\.dot),
                          totalDots: perDot.count,
                          residualPoints: resFinal,
                          rejectedDotIndices: rejected)
    }

    /// Shared normal-equation solve: stacks two rows per entry into
    /// `(AᵀA)·t = Aᵀb` and inverts. Extracted so `fit` and `fitRobust`
    /// can't drift apart.
    private static func solveT(
        rows: [(r: Double, q: Double, target: simd_double2, dot: Int)]
    ) -> simd_double3? {
        var ata = simd_double3x3()
        var atb = simd_double3()
        for row in rows {
            let rowX = simd_double3(1, 0, row.r)
            let rowY = simd_double3(0, 1, row.q)
            ata.columns.0 += rowX * rowX.x + rowY * rowY.x
            ata.columns.1 += rowX * rowX.y + rowY * rowY.y
            ata.columns.2 += rowX * rowX.z + rowY * rowY.z
            atb += rowX * row.target.x + rowY * row.target.y
        }
        guard abs(ata.determinant) > 1e-9 else { return nil }
        return ata.inverse * atb
    }

    /// Median of an unsorted array; nil when empty.
    static func median(_ xs: [Double]) -> Double? {
        let v = xs.filter(\.isFinite).sorted()
        guard !v.isEmpty else { return nil }
        let m = v.count / 2
        return v.count % 2 == 1 ? v[m] : (v[m - 1] + v[m]) * 0.5
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
