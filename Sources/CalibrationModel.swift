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
    /// Fitted `(tx, ty, tz)` from `ScreenMapper.fit`.
    let translation: simd_double3
    /// 9 calibration target positions used during fit.
    let targets: [simd_double2]
    /// Per-target mean residual `actual - global_predicted` from the fit.
    /// Same indexing as `targets`.
    let offsets: [simd_double2]

    /// Project a unit gaze (camera frame) to a screen-relative point.
    ///
    /// PIPELINE.md §Stage 5 prescribes a "median of global and nearest-local"
    /// blend. A two-element median is just the mean, so we average the
    /// global prediction with `global + offset[nearest target]`.
    func predict(gazeCam: simd_double3) -> simd_double2 {
        let pGlobal = ScreenMapper.project(gazeCam: gazeCam,
                                            translation: translation)
        guard pGlobal.x.isFinite, pGlobal.y.isFinite else { return pGlobal }
        // "Nearest local" = the calibration target whose position is
        // closest to the current global prediction. Conceptually we apply
        // the region-specific bias correction for the part of the screen
        // the user appears to be looking at.
        var bestIdx = 0
        var bestDist = Double.infinity
        for i in 0..<targets.count {
            let d = simd_distance(targets[i], pGlobal)
            if d < bestDist { bestDist = d; bestIdx = i }
        }
        let pLocal = pGlobal + offsets[bestIdx]
        return (pGlobal + pLocal) * 0.5
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
        var allSamples: [(gaze: simd_double3, target: simd_double2)] = []
        for (i, ds) in samples.enumerated() {
            for g in ds { allSamples.append((g, targets[i])) }
        }
        guard let t = ScreenMapper.fit(samples: allSamples) else {
            phase = .failed(allSamples.isEmpty
                ? "No gaze samples (Stage 4 model not loaded?)"
                : "Fit failed — need more diverse samples")
            return
        }
        // Per-target offsets: mean residual of (target - global_pred) over
        // that target's own samples.
        var offsets: [simd_double2] = []
        offsets.reserveCapacity(targets.count)
        for (i, ds) in samples.enumerated() {
            guard !ds.isEmpty else { offsets.append(.zero); continue }
            var sum = simd_double2(0, 0)
            var n = 0
            for g in ds {
                let p = ScreenMapper.project(gazeCam: g, translation: t)
                if p.x.isFinite && p.y.isFinite {
                    sum += targets[i] - p
                    n += 1
                }
            }
            offsets.append(n > 0 ? sum / Double(n) : .zero)
        }
        self.result = CalibrationModel(
            translation: t,
            targets: targets,
            offsets: offsets
        )
        phase = .complete
    }
}
