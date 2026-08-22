import Foundation
import simd

/// Screen-points ↔ degrees-of-visual-angle conversion.
///
/// Every experiment reports error in both px (screen points) and degrees, so
/// the conversion lives in one place rather than being re-derived per model.
///
/// **The distance used is virtual, not measured.** Stage 5 fits the screen
/// centre's pose `t = (tx, ty, tz)` in camera coords with the gaze ray
/// emanating from the camera origin (see `ScreenMapper`), so `|tz|` is the
/// fitted eye-to-screen distance *in screen points*. It is self-consistent
/// with the calibration that produced the prediction being scored, which is
/// what matters for comparing conditions within a session — but it is not a
/// physically measured viewing distance, so degrees are not comparable across
/// participants whose calibration fits differ wildly.
enum VisualAngle {

    /// Fitted eye-to-screen distance in screen points for a calibration.
    /// Negative `tz` just means the screen sits behind the camera origin
    /// along the optical axis; only the magnitude matters here.
    static func distancePoints(for calibration: CalibrationModel) -> Double {
        abs(calibration.translation.z)
    }

    /// Convert an on-screen error magnitude to degrees of visual angle.
    /// Returns NaN when the distance is unusable, so it flows through the
    /// CSV formatters as an empty cell instead of a misleading 0.
    static func degrees(points: Double, distancePoints d: Double) -> Double {
        guard points.isFinite, d.isFinite, d > 0 else { return .nan }
        return atan(points / d) * 180.0 / .pi
    }

    /// Mean of a set of point-errors expressed in degrees. Converts each
    /// error individually before averaging — `atan` is non-linear, so
    /// converting the mean would not give the mean of the conversions.
    static func meanDegrees(pointErrors: [Double], distancePoints d: Double) -> Double {
        let vals = pointErrors
            .map { degrees(points: $0, distancePoints: d) }
            .filter { $0.isFinite }
        guard !vals.isEmpty else { return .nan }
        return vals.reduce(0, +) / Double(vals.count)
    }
}

/// Display formatting for metrics that are legitimately undefined.
///
/// A run with no scored samples (face lost, cancelled early, no dot confirmed)
/// produces NaN for every statistic, which `String(format: "%.2f")` renders as
/// "nan" — that reads as a crash rather than as "not measured". These render an
/// em dash instead. The CSV writers already emit an empty cell for the same
/// case; this is the on-screen counterpart.
enum MetricFormat {
    static let undefined = "—"

    /// e.g. "1.24°", or "—".
    static func degrees(_ x: Double, places: Int = 2) -> String {
        x.isFinite ? String(format: "%.\(places)f°", x) : undefined
    }

    /// e.g. "21.7 pt", or "—".
    static func points(_ x: Double, places: Int = 1) -> String {
        x.isFinite ? String(format: "%.\(places)f pt", x) : undefined
    }

    /// e.g. "83%", or "—". Takes a fraction in [0, 1].
    static func percent(_ fraction: Double, places: Int = 0) -> String {
        fraction.isFinite
            ? String(format: "%.\(places)f%%", fraction * 100.0) : undefined
    }

    /// e.g. "2.85 s", or "—".
    static func seconds(_ x: Double, places: Int = 2) -> String {
        x.isFinite ? String(format: "%.\(places)f s", x) : undefined
    }

    /// A bare number with no unit, e.g. "17.5", or "—".
    static func number(_ x: Double, places: Int = 1) -> String {
        x.isFinite ? String(format: "%.\(places)f", x) : undefined
    }

    /// "21.7 pt   1.77°" — the paired form used by the detail rows, with
    /// each half independently falling back to an em dash.
    static func pointsAndDegrees(_ pt: Double, _ deg: Double) -> String {
        "\(points(pt))   \(degrees(deg))"
    }
}

/// Scatter statistics for a cloud of predicted points around their own
/// centroid (precision) and around a known target (accuracy).
///
/// Kept separate from the experiment models because Experiment 2 (fixation
/// stability) and the calibration validation both need exactly this, computed
/// the same way, or their numbers can't be compared to each other.
struct GazeScatter {
    /// Centroid of the samples.
    let mean: simd_double2
    /// Mean Euclidean distance from each sample to `target`. This is the
    /// accuracy measure: "mean deviation from the circle centre".
    let meanDeviation: Double
    /// Sample standard deviation of the distances to `mean` — the spread of
    /// the cloud about its own centre, independent of any systematic offset.
    let standardDeviation: Double
    /// Root-mean-square distance from each sample to `mean`. The standard
    /// eye-tracking "precision" figure; for a 2-D cloud this is the
    /// quadrature sum of the per-axis standard deviations.
    let rms: Double
    /// Euclidean distance from `mean` to `target` — the systematic component
    /// of the error, with jitter averaged out.
    let bias: Double
    let count: Int

    /// Compute scatter of `samples` relative to a known `target`. Returns
    /// nil for an empty input; `standardDeviation` is NaN for a single
    /// sample (no degrees of freedom).
    static func compute(samples: [simd_double2],
                        target: simd_double2) -> GazeScatter? {
        guard !samples.isEmpty else { return nil }
        let n = Double(samples.count)
        var sum = simd_double2(0, 0)
        for s in samples { sum += s }
        let mean = sum / n

        var devSum = 0.0        // Σ |s - target|
        var sqSum = 0.0         // Σ |s - mean|²
        for s in samples {
            devSum += simd_distance(s, target)
            let d = simd_distance(s, mean)
            sqSum += d * d
        }
        let rms = sqrt(sqSum / n)

        // Sample SD of the radial distances about the centroid. Uses the
        // n-1 denominator so a short fixation isn't reported as tighter
        // than it is.
        let sd: Double
        if samples.count > 1 {
            let dists = samples.map { simd_distance($0, mean) }
            let dMean = dists.reduce(0, +) / n
            let varSum = dists.reduce(0.0) { $0 + ($1 - dMean) * ($1 - dMean) }
            sd = sqrt(varSum / (n - 1))
        } else {
            sd = .nan
        }

        return GazeScatter(
            mean: mean,
            meanDeviation: devSum / n,
            standardDeviation: sd,
            rms: rms,
            bias: simd_distance(mean, target),
            count: samples.count
        )
    }
}
