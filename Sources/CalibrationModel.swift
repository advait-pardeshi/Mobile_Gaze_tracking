import Foundation
import simd
import QuartzCore

/// Persistent result of a Stage 5 calibration session: the global screen
/// translation `t` plus per-target residual offsets used as local
/// corrections.
///
/// Coordinates throughout are screen-center-relative **points** (UIKit
/// logical units; +x right, +y down).
struct CalibrationModel {

    /// How the per-target residual offsets are blended into a prediction.
    enum LocalCorrection: String, CaseIterable {
        /// Original behaviour: apply the offset of whichever target is
        /// closest to the global prediction. Piecewise-constant, so the
        /// correction jumps discontinuously across the Voronoi boundaries
        /// between targets — a gaze drifting slowly across such a boundary
        /// makes the predicted point step sideways by the difference between
        /// two neighbouring offsets, with no motion of the eye behind it.
        case nearestNeighbour
        /// Gaussian-RBF weighting over every valid target offset. Same
        /// stored data, but the correction field is now smooth everywhere,
        /// and every dot contributes in proportion to how near it is.
        case gaussianRBF
    }

    /// The flag. Flip to `.nearestNeighbour` to restore the old path
    /// wholesale; `Tools/CalibrationReplay` scores both against recorded runs.
    static var defaultLocalCorrection: LocalCorrection = .gaussianRBF

    /// Default kernel width as a multiple of the median nearest-neighbour
    /// spacing of the target grid.
    ///
    /// Expressed relative to spacing rather than in absolute points so it
    /// holds across screen sizes and target layouts. At 1.0 a target's
    /// nearest neighbour sits exactly one σ away and carries e^-0.5 ≈ 0.61 of
    /// its weight — enough overlap to be smooth, not so much that all nine
    /// offsets average into a single global constant.
    static var defaultSigmaSpacingMultiplier: Double = 1.0

    /// Fitted `(tx, ty, tz)` from `ScreenMapper.fit`.
    let translation: simd_double3
    /// 9 calibration target positions used during fit.
    let targets: [simd_double2]
    /// Per-target mean residual `actual - global_predicted` from the fit.
    /// Same indexing as `targets`. Dots rejected by the robust fit carry
    /// `.zero`, so `predict` falls back to the global projection in their
    /// region rather than applying a correction derived from bad samples.
    let offsets: [simd_double2]
    /// Diagnostics from the robust fit that produced `translation`. Nil only
    /// for models restored from an older on-disk representation.
    var fitQuality: ScreenMapper.FitQuality? = nil

    /// Which blending rule `predict` uses. Stored per-model rather than read
    /// from the static flag at call time, so a replay harness can hold both
    /// variants of the *same* fit side by side without mutating global state.
    var localCorrection: LocalCorrection = CalibrationModel.defaultLocalCorrection
    /// Gaussian kernel width in screen points. `nil` derives it from the
    /// target spacing — see `effectiveSigmaPoints`.
    var rbfSigmaPoints: Double? = nil

    /// Kernel width actually used, in screen points.
    var effectiveSigmaPoints: Double {
        rbfSigmaPoints ?? Self.defaultSigmaPoints(targets: targets)
    }

    /// Indices whose offset carries real information.
    ///
    /// A dot rejected by the robust fit has `offset == .zero`, but that zero
    /// means *"no measurement here"*, not *"the correction here is zero"*.
    /// Nearest-neighbour conflates the two harmlessly, because a rejected
    /// dot only ever affects its own cell. A weighted mean cannot afford the
    /// same sloppiness: including a rejected dot as a zero would drag the
    /// correction toward zero across its whole neighbourhood, corrupting
    /// dots that were fine. So the RBF path skips them and lets the valid
    /// neighbours extrapolate over the gap instead.
    var validOffsetIndices: [Int] {
        let n = min(targets.count, offsets.count)
        if let rejected = fitQuality?.rejectedDotIndices {
            let r = Set(rejected)
            return (0..<n).filter { !r.contains($0) }
        }
        // Older models carry no fit diagnostics; an exactly-zero offset is
        // the only remaining signal, and a genuinely-zero residual is a
        // measure-zero coincidence.
        return (0..<n).filter { offsets[$0] != .zero }
    }

    /// Project a unit gaze (camera frame) to a screen-relative point.
    ///
    /// PIPELINE.md §Stage 5 prescribes a "median of global and nearest-local"
    /// blend. A two-element median is just the mean, so the local correction
    /// enters at half weight. That factor is preserved verbatim across both
    /// `LocalCorrection` modes — the only thing this branch changes is *how
    /// the correction at a point is derived from the nine stored offsets*,
    /// not how strongly it is applied.
    func predict(gazeCam: simd_double3) -> simd_double2 {
        let pGlobal = ScreenMapper.project(gazeCam: gazeCam,
                                            translation: translation)
        guard pGlobal.x.isFinite, pGlobal.y.isFinite else { return pGlobal }
        guard let correction = localCorrectionOffset(at: pGlobal) else {
            return pGlobal
        }
        return pGlobal + correction * 0.5
    }

    /// The interpolated residual correction at a screen point, or nil when
    /// there is no usable offset to apply (in which case `predict` falls back
    /// to the bare global projection).
    ///
    /// Exposed rather than inlined so the replay harness can map the
    /// correction field itself, independently of any gaze data.
    func localCorrectionOffset(at p: simd_double2) -> simd_double2? {
        switch localCorrection {
        case .nearestNeighbour:
            return nearestOffset(at: p)
        case .gaussianRBF:
            return rbfOffset(at: p)
        }
    }

    /// Original path, unchanged — including the detail that it searches
    /// *all* targets, rejected ones included, so a rejected dot still
    /// produces a zero-correction region exactly as it always did.
    private func nearestOffset(at p: simd_double2) -> simd_double2? {
        guard !targets.isEmpty, !offsets.isEmpty else { return nil }
        var bestIdx = 0
        var bestDist = Double.infinity
        for i in 0..<min(targets.count, offsets.count) {
            let d = simd_distance(targets[i], p)
            if d < bestDist { bestDist = d; bestIdx = i }
        }
        return offsets[bestIdx]
    }

    /// Gaussian-RBF weighted mean of the valid offsets:
    ///
    ///     w_i = exp(−‖tᵢ − p‖² / 2σ²)
    ///     c(p) = Σ wᵢ·oᵢ / Σ wᵢ
    ///
    /// Normalised (Shepard-style) rather than a raw RBF sum, so the
    /// correction stays bounded by the convex hull of the stored offsets and
    /// can never amplify one — a far-off-screen prediction degrades to the
    /// nearest offset instead of diverging.
    ///
    /// As σ → 0 this converges to nearest-neighbour *over the valid offsets*,
    /// and as σ → ∞ to their plain mean — a single constant bias correction.
    /// The tunable slides between those two. Note the σ → 0 limit is not
    /// quite `nearestOffset`: that one also considers rejected dots, so the
    /// two differ wherever a dot was rejected.
    private func rbfOffset(at p: simd_double2) -> simd_double2? {
        let valid = validOffsetIndices
        guard !valid.isEmpty else { return nil }

        let sigma = max(1e-6, effectiveSigmaPoints)
        let twoSigmaSq = 2.0 * sigma * sigma

        // Pass 1: the nearest valid anchor. Its squared distance is
        // subtracted from every exponent in pass 2, which makes the nearest
        // weight exactly 1 and every other ≤ 1. Mathematically a no-op — the
        // common factor cancels in the normalised mean — but it removes the
        // underflow that would otherwise zero the entire weight set for a
        // prediction a few thousand points off screen.
        var nearestIdx = valid[0]
        var nearestD2 = Double.infinity
        for i in valid {
            let d2 = simd_length_squared(targets[i] - p)
            if d2 < nearestD2 { nearestD2 = d2; nearestIdx = i }
        }
        guard nearestD2.isFinite else { return nil }

        var weightSum = 0.0
        var accum = simd_double2(0, 0)
        for i in valid {
            let d2 = simd_length_squared(targets[i] - p)
            let w = exp(-(d2 - nearestD2) / twoSigmaSq)
            weightSum += w
            accum += offsets[i] * w
        }
        // Unreachable given the shift above (the nearest term contributes
        // exactly 1), but a normalised mean must never divide by zero.
        guard weightSum > 0, weightSum.isFinite else {
            return offsets[nearestIdx]
        }
        return accum / weightSum
    }

    /// Per-target mean residual `target − global_prediction`, over that
    /// target's own gaze samples.
    ///
    /// Extracted from `CalibrationController.finish` so the replay harness
    /// builds its models through exactly this code rather than a copy of it.
    /// A second implementation would silently drift from the shipping one,
    /// and a replay that scores a drifted model is worse than no replay.
    ///
    /// Rejected dots get `.zero`: their samples are precisely the ones the
    /// robust fit just decided not to trust.
    static func residualOffsets(
        perDot: [(gazes: [simd_double3], target: simd_double2)],
        translation t: simd_double3,
        rejected: Set<Int>
    ) -> [simd_double2] {
        var offsets: [simd_double2] = []
        offsets.reserveCapacity(perDot.count)
        for (i, dot) in perDot.enumerated() {
            guard !dot.gazes.isEmpty, !rejected.contains(i) else {
                offsets.append(.zero)
                continue
            }
            var sum = simd_double2(0, 0)
            var n = 0
            for g in dot.gazes {
                let p = ScreenMapper.project(gazeCam: g, translation: t)
                if p.x.isFinite && p.y.isFinite {
                    sum += dot.target - p
                    n += 1
                }
            }
            offsets.append(n > 0 ? sum / Double(n) : .zero)
        }
        return offsets
    }

    /// Median nearest-neighbour spacing of the target grid, scaled by
    /// `defaultSigmaSpacingMultiplier`.
    ///
    /// Median rather than mean so an unusual layout (or a single outlying
    /// target) can't stretch the kernel across the whole screen.
    static func defaultSigmaPoints(targets: [simd_double2]) -> Double {
        guard targets.count > 1 else { return 1.0 }
        var nn: [Double] = []
        nn.reserveCapacity(targets.count)
        for i in targets.indices {
            var best = Double.infinity
            for j in targets.indices where j != i {
                best = min(best, simd_distance(targets[i], targets[j]))
            }
            if best.isFinite { nn.append(best) }
        }
        guard let m = ScreenMapper.median(nn), m > 0 else { return 1.0 }
        return m * defaultSigmaSpacingMultiplier
    }
}

/// Drives the 9-dot calibration UI: dwell → capture per dot, advance, fit.
///
/// Per PIPELINE.md the user stares at each dot for ~3 s; we split that into
/// a 1 s **dwell** (let the user fixate, no samples taken) and a 2 s
/// **capture** window. Phases are observable so the SwiftUI overlay can
/// retint the dot during capture.
@MainActor
final class CalibrationController: ObservableObject {

    enum Phase: Equatable {
        case idle
        case dwelling
        case capturing
        case complete
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var dotIndex: Int = 0
    /// Progress within the *current* dot, in `[0, 1]` over `pre + capture`.
    @Published private(set) var dotProgress: Double = 0
    /// Set on completion; nil while running, on cancel, or on failure.
    @Published private(set) var result: CalibrationModel?

    let targets: [simd_double2]
    let preDuration: CFTimeInterval
    let captureDuration: CFTimeInterval

    private var samples: [[simd_double3]]
    private var dotStart: CFTimeInterval = 0
    private var timer: Timer?

    init(targets: [simd_double2],
         preDuration: CFTimeInterval = 1.0,
         captureDuration: CFTimeInterval = 2.0) {
        self.targets = targets
        self.preDuration = preDuration
        self.captureDuration = captureDuration
        self.samples = Array(repeating: [], count: targets.count)
    }

    deinit { timer?.invalidate() }

    /// Begin the run from dot 0. Idempotent if already running.
    func start() {
        guard !targets.isEmpty else {
            phase = .failed("No targets")
            return
        }
        timer?.invalidate()
        dotIndex = 0
        dotStart = CACurrentMediaTime()
        dotProgress = 0
        phase = .dwelling
        let t = Timer(timeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
        RunLoop.main.add(t, forMode: .common)
        self.timer = t
    }

    /// Abort. Safe to call from any phase.
    func cancel() {
        timer?.invalidate()
        timer = nil
        phase = .idle
        dotProgress = 0
        result = nil
    }

    /// Feed a gaze sample. Only retained during the **capture** window of
    /// the active dot; ignored otherwise.
    func ingest(gazeCam: simd_double3) {
        guard phase == .capturing else { return }
        guard dotIndex < samples.count else { return }
        samples[dotIndex].append(gazeCam)
    }

    var currentTarget: simd_double2 {
        guard dotIndex < targets.count else { return .zero }
        return targets[dotIndex]
    }

    private func tick() {
        let elapsed = CACurrentMediaTime() - dotStart
        let total = preDuration + captureDuration
        dotProgress = min(1.0, elapsed / total)
        switch phase {
        case .dwelling where elapsed >= preDuration:
            phase = .capturing
        case .capturing where elapsed >= total:
            advance()
        default:
            break
        }
    }

    private func advance() {
        let next = dotIndex + 1
        if next < targets.count {
            dotIndex = next
            dotStart = CACurrentMediaTime()
            dotProgress = 0
            phase = .dwelling
        } else {
            timer?.invalidate()
            timer = nil
            finish()
        }
    }

    private func finish() {
        let totalSamples = samples.reduce(0) { $0 + $1.count }
        guard totalSamples > 0 else {
            phase = .failed("No gaze samples (Stage 4 model not loaded?)")
            return
        }

        // Robust fit rather than `ScreenMapper.fit(samples:)`: one dot the
        // participant never actually fixated otherwise biases `t` — and
        // therefore *every* prediction on screen — by a constant offset that
        // no downstream stage can recover from.
        let perDot = samples.enumerated().map {
            (gazes: $0.element, target: targets[$0.offset])
        }
        guard let q = ScreenMapper.fitRobust(perDot: perDot) else {
            phase = .failed("Fit failed — too few usable dots")
            return
        }
        let t = q.translation

        // Reject the whole fit if what survived still doesn't reproduce the
        // targets. This is the loud failure that used to be a silent bad `t`.
        guard q.medianResidualPoints <= Self.maxAcceptableResidualPoints else {
            phase = .failed(String(
                format: "Fit rejected — median residual %.0f pt (limit %.0f)",
                q.medianResidualPoints, Self.maxAcceptableResidualPoints))
            return
        }

        // Per-target offsets: mean residual of (target - global_pred) over
        // that target's own samples.
        let offsets = CalibrationModel.residualOffsets(
            perDot: perDot,
            translation: t,
            rejected: Set(q.rejectedDotIndices))
        if !q.rejectedDotIndices.isEmpty {
            print("[Gaze] Stage 5: rejected calibration dots \(q.rejectedDotIndices.sorted()) "
                  + String(format: "(median residual %.1f pt)", q.medianResidualPoints))
        }
        self.result = CalibrationModel(
            translation: t,
            targets: targets,
            offsets: offsets,
            fitQuality: q
        )
        phase = .complete
    }

    /// Median per-dot residual above which the fit is thrown away outright.
    /// Roughly a fifth of a phone's short edge — a fit that can't reproduce
    /// its own calibration dots to better than this reproduces nothing.
    static let maxAcceptableResidualPoints: Double = 80.0
}
