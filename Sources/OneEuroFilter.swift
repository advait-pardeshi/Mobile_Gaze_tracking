import Foundation
import simd

/// The 1€ filter (Casiez, Roussel & Vogel, CHI 2012).
///
/// An adaptive exponential smoother: the cutoff frequency rises with the
/// observed rate of change, so a held signal is smoothed hard while a fast
/// one passes through with little lag. That trade is the whole reason it
/// replaces the constant-velocity Kalman here — the Kalman's lag is fixed by
/// its noise ratio, so tuning it quiet enough for a fixation made every
/// saccade visibly late.
///
///     τ      = 1 / (2π · f_c)
///     α(dt)  = 1 / (1 + τ/dt)
///     dx̂     = lowpass(dx,  α(d_cutoff))
///     f_c    = minCutoff + β · |dx̂|
///     x̂      = lowpass(x,   α(f_c))
///
/// `minCutoff` sets the noise floor (lower = smoother at rest); `beta` sets
/// how quickly the filter gets out of the way (higher = less lag on fast
/// motion, more jitter passed through).
struct OneEuroFilter {

    let minCutoff: Double
    let beta: Double
    let dCutoff: Double

    private var xHat: Double = 0
    private var dxHat: Double = 0
    private var initialized = false

    init(minCutoff: Double, beta: Double, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    var value: Double? { initialized ? xHat : nil }

    mutating func reset() {
        initialized = false
        xHat = 0
        dxHat = 0
    }

    /// Feed one sample. `dt` is seconds since the previous call; it is
    /// clamped so a paused-then-resumed pipeline can't produce α ≈ 1 (no
    /// smoothing) or α ≈ 0 (a frozen filter).
    mutating func update(_ x: Double, dt: TimeInterval) -> Double {
        guard x.isFinite else { return xHat }
        guard initialized else {
            xHat = x
            dxHat = 0
            initialized = true
            return x
        }
        let dtc = OneEuro.clampDt(dt)
        let dx = (x - xHat) / dtc
        dxHat = OneEuro.lowpass(dx, previous: dxHat,
                                alpha: OneEuro.alpha(cutoff: dCutoff, dt: dtc))
        let cutoff = minCutoff + beta * abs(dxHat)
        xHat = OneEuro.lowpass(x, previous: xHat,
                               alpha: OneEuro.alpha(cutoff: cutoff, dt: dtc))
        return xHat
    }
}

/// 1€ filter over a 3-vector.
///
/// The cutoff is driven by the **speed magnitude** of the vector rather than
/// by each component independently. Per-component cutoffs would let x, y and
/// z be smoothed by different amounts on the same frame, which bends the
/// direction of the vector — wrong for a position that has to stay
/// geometrically coherent.
struct OneEuroVector3 {

    let minCutoff: Double
    let beta: Double
    let dCutoff: Double

    private var xHat = simd_double3()
    private var dxHat = simd_double3()
    private var initialized = false

    init(minCutoff: Double, beta: Double, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    var value: simd_double3? { initialized ? xHat : nil }

    mutating func reset() {
        initialized = false
        xHat = simd_double3()
        dxHat = simd_double3()
    }

    mutating func update(_ x: simd_double3, dt: TimeInterval) -> simd_double3 {
        guard x.x.isFinite, x.y.isFinite, x.z.isFinite else { return xHat }
        guard initialized else {
            xHat = x
            dxHat = simd_double3()
            initialized = true
            return x
        }
        let dtc = OneEuro.clampDt(dt)
        let dx = (x - xHat) / dtc
        let ad = OneEuro.alpha(cutoff: dCutoff, dt: dtc)
        dxHat = dxHat + (dx - dxHat) * ad
        let cutoff = minCutoff + beta * simd_length(dxHat)
        let a = OneEuro.alpha(cutoff: cutoff, dt: dtc)
        xHat = xHat + (x - xHat) * a
        return xHat
    }
}

/// 1€ filter over a rotation, expressed as a unit quaternion.
///
/// Filtering Euler angles instead would be wrong twice over: the three angles
/// are not independent (smoothing them separately produces a rotation that is
/// not the smoothed rotation), and `atan2`/`asin` wrap, so a yaw crossing ±180°
/// injects a full-scale step into the filter. Here the "derivative" is the
/// geodesic angle between consecutive rotations in radians/second, and the
/// low-pass step is a SLERP by `α` toward the new sample — both wrap-free and
/// both intrinsic to the rotation manifold.
struct OneEuroQuaternion {

    let minCutoff: Double
    let beta: Double
    let dCutoff: Double

    private var qHat = simd_quatd(ix: 0, iy: 0, iz: 0, r: 1)
    private var dAngleHat: Double = 0
    private var initialized = false

    init(minCutoff: Double, beta: Double, dCutoff: Double = 1.0) {
        self.minCutoff = minCutoff
        self.beta = beta
        self.dCutoff = dCutoff
    }

    var value: simd_quatd? { initialized ? qHat : nil }

    mutating func reset() {
        initialized = false
        qHat = simd_quatd(ix: 0, iy: 0, iz: 0, r: 1)
        dAngleHat = 0
    }

    mutating func update(_ qIn: simd_quatd, dt: TimeInterval) -> simd_quatd {
        var q = qIn
        guard q.vector.x.isFinite, q.vector.y.isFinite,
              q.vector.z.isFinite, q.vector.w.isFinite,
              simd_length(q.vector) > 1e-9 else { return qHat }
        q = q.normalized

        guard initialized else {
            qHat = q
            dAngleHat = 0
            initialized = true
            return q
        }

        // `q` and `-q` are the same rotation. Pick the hemisphere nearer the
        // running estimate, or both the angle and the SLERP take the long way
        // around and the filter fights a 360° phantom rotation.
        if simd_dot(q.vector, qHat.vector) < 0 {
            q = simd_quatd(vector: -q.vector)
        }

        let dtc = OneEuro.clampDt(dt)
        // Geodesic angle between the two rotations, radians.
        let dot = min(1.0, max(-1.0, simd_dot(q.vector, qHat.vector)))
        let angle = 2.0 * acos(dot)
        let speed = angle / dtc
        dAngleHat = OneEuro.lowpass(speed, previous: dAngleHat,
                                    alpha: OneEuro.alpha(cutoff: dCutoff, dt: dtc))

        let cutoff = minCutoff + beta * abs(dAngleHat)
        let a = OneEuro.alpha(cutoff: cutoff, dt: dtc)
        qHat = simd_slerp(qHat, q, a).normalized
        return qHat
    }
}

/// Shared scalar helpers, so the three filters can't drift apart.
enum OneEuro {
    /// Bounded to [1/240 s, 0.2 s]: below that a timestamp collision would
    /// divide by ~0, above it a resumed pipeline would effectively bypass the
    /// filter on its first frame.
    static func clampDt(_ dt: TimeInterval) -> Double {
        guard dt.isFinite, dt > 0 else { return 1.0 / 30.0 }
        return min(0.2, max(1.0 / 240.0, dt))
    }

    static func alpha(cutoff: Double, dt: Double) -> Double {
        let tau = 1.0 / (2.0 * Double.pi * max(1e-6, cutoff))
        return 1.0 / (1.0 + tau / dt)
    }

    static func lowpass(_ x: Double, previous: Double, alpha a: Double) -> Double {
        previous + (x - previous) * a
    }
}

extension simd_quatd {
    var normalized: simd_quatd {
        let n = simd_length(vector)
        guard n > 1e-12 else { return simd_quatd(ix: 0, iy: 0, iz: 0, r: 1) }
        return simd_quatd(vector: vector / n)
    }
}
