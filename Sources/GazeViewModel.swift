import Foundation
import AVFoundation
import CoreML
@preconcurrency import MediaPipeTasksVision
import UIKit
import simd

/// Thread-safe `(timestampMs → CVPixelBuffer)` cache.
/// The camera queue stores buffers; the main actor pops them when the
/// matching MediaPipe result arrives. Entries older than the requested
/// timestamp are evicted on read since they're no longer useful.
final class PendingBufferCache {
    private let lock = NSLock()
    private var buffers: [Int: CVPixelBuffer] = [:]
    private let capacity: Int

    init(capacity: Int = 6) { self.capacity = capacity }

    func store(_ pb: CVPixelBuffer, timestampMs: Int) {
        lock.lock()
        defer { lock.unlock() }
        buffers[timestampMs] = pb
        while buffers.count > capacity, let oldest = buffers.keys.min() {
            buffers.removeValue(forKey: oldest)
        }
    }

    func take(timestampMs: Int) -> CVPixelBuffer? {
        lock.lock()
        defer { lock.unlock() }
        let pb = buffers.removeValue(forKey: timestampMs)
        for k in buffers.keys where k < timestampMs {
            buffers.removeValue(forKey: k)
        }
        return pb
    }
}

@MainActor
final class GazeViewModel: ObservableObject {
    @Published var landmarks: [NormalizedLandmark] = []
    @Published var faceCount: Int = 0
    @Published var imageSize: CGSize = .zero
    @Published var fps: Double = 0
    @Published var headPose: HeadPose?
    /// Diagnostic: most recent reason head-pose estimation failed (nil when ok).
    @Published var headPoseFailureReason: String?
    /// Diagnostic: pinhole intrinsics summary, refreshed on first frame.
    @Published var intrinsicsSummary: String?

    /// Stage 3 visualization: 200×50 head-pose-normalized eye strip.
    @Published var normalizedEyes: UIImage?
    /// Most recent eye-strip tensor (50·200·3 floats in [0,1]). Currently
    /// unused by Stage 4 (we switched to a face-crop CNN); kept around as
    /// diagnostic state and for any future eye-stream ML.
    ///
    /// Deliberately **not** `@Published`: nothing observes it, and republishing
    /// a 30 000-element array on every frame woke every SwiftUI view that
    /// observes this object, per frame, for nobody.
    private(set) var normalizedEyesTensor: [Float] = []
    /// Stage 4 input: 224×224 head-pose-normalized + horizontally-mirrored
    /// face crop (the input ETH-XGaze ResNet18 was trained on).
    @Published var normalizedFace: UIImage?

    /// Stage 4 output: gaze direction in camera space (or nil if no model is
    /// bundled / the current frame had no usable pose).
    @Published var gaze: GazeEstimator.Estimate?
    /// Midpoint between the two eye centers in camera coords (mm). Used as
    /// the anchor for the gaze ray overlay.
    @Published var gazeOriginCam: simd_double3?

    /// Stage 5: live size of the calibration / display surface (geometry
    /// reader value), in points. Calibration targets and the final gaze dot
    /// are positioned in this coordinate space.
    @Published var screenSize: CGSize = .zero
    /// Active calibration session, or nil when not calibrating.
    @Published var calibrationController: CalibrationController?
    /// Most recent fitted calibration. Persists across sessions until the
    /// user re-calibrates.
    @Published var calibration: CalibrationModel?
    /// Live gaze dot in absolute view-space points, or nil when no
    /// prediction is available (no calibration yet, or no Stage 4 gaze).
    @Published var gazeScreenPoint: CGPoint?

    /// Active accuracy-test session, or nil. Only meaningful when
    /// `calibration` is set (the test feeds calibrated predictions).
    @Published var accuracyController: AccuracyTestController?
    /// Most recent accuracy-test result. Set when a run completes; cleared
    /// when the user dismisses the results screen or starts a new run.
    @Published var accuracyResult: AccuracyTestResult?

    /// Experiment 1 (grid-focus) — active run controller, or nil.
    @Published var gridExperimentController: GridExperimentController?
    /// Experiment 1 — most recent result.
    @Published var gridExperimentResult: GridExperimentResult?

    /// Calibration validation — active run, or nil. Started automatically as
    /// soon as a calibration fit completes, so a fit is never used by an
    /// experiment before it has been checked.
    @Published var validationController: CalibrationValidationController?
    /// Calibration validation — most recent result. Cleared when the operator
    /// dismisses the verdict screen.
    @Published var validationResult: CalibrationValidationResult?
    /// Verdict of the most recent validation of the *current* calibration, kept
    /// after the results screen is dismissed so the HUD can keep showing
    /// whether the active calibration was ever checked. Reset by a new
    /// calibration run.
    @Published var lastValidationVerdict: ValidationVerdict?
    /// Mean error in degrees from that validation, for the same HUD line.
    @Published var lastValidationMeanDegrees: Double = .nan

    /// Experiment 2 (fixation stability) — active controller, or nil.
    @Published var fixationController: FixationStabilityController?
    /// Experiment 2 — most recent result.
    @Published var fixationResult: FixationStabilityResult?

    /// Experiment 3 (image-based communication task) — active controller.
    @Published var commTaskController: CommunicationTaskController?
    /// Experiment 3 — most recent result.
    @Published var commTaskResult: CommunicationTaskResult?
    /// Experiment 3 — last finished run. Not `@Published`: it drives no UI,
    /// because Experiment 3 deliberately has no results screen. Held only so
    /// the composed sentence can still be replayed after the run.
    private(set) var lastCommTaskResult: CommunicationTaskResult?

    /// Hot-swap model state: short status string surfaced in the HUD / fetch
    /// sheet. Examples: "fetching… 2.1 MB", "loaded p07_GazeNet (12 files)",
    /// "fetch failed: manifest HTTP 404". Cleared after success.
    @Published var modelFetchStatus: String?
    /// True while a fetch is in-flight so the UI can disable the button.
    @Published var modelFetchInFlight: Bool = false
    /// Human-readable description of the currently-loaded weights.
    /// Mirrors `GazeEstimator.modelSource` so SwiftUI can observe it.
    @Published var modelSource: String = "bundled"
    /// Set to the swapped-in model name when a fetch+compile+swap fully
    /// succeeds. The fetch sheet observes this to raise a success alert,
    /// then clears it back to nil on dismissal.
    @Published var modelFetchSuccess: String?

    // Nonisolated: read from the camera queue and the pipeline queue as well
    // as the main actor. `intrinsics` is written once during `configure()`
    // and read thereafter (the file documents it as single-writer).
    nonisolated let cameraManager = CameraManager()
    // Nonisolated so the camera queue can call detectAsync synchronously
    // without an actor hop (which would let Swift reorder frame submission
    // and trip MediaPipe's monotonic-timestamp requirement).
    nonisolated private let landmarkerService: FaceLandmarkerService?

    /// Stages 2–4. Confined to `processQueue` — see `GazePipeline`.
    nonisolated private let pipeline = GazePipeline()

    /// The serial queue that owns `pipeline`. `.userInitiated` rather than
    /// `.userInteractive`: this work must not compete with the main thread
    /// it was moved off.
    nonisolated private let processQueue = DispatchQueue(
        label: "gaze.pipeline", qos: .userInitiated)

    /// Raised for the whole duration of a frame — worker stages *and* the
    /// main-actor publish. While it is up, arriving frames are dropped rather
    /// than queued, so latency stays bounded by one frame's work instead of
    /// growing without limit behind a slow one.
    nonisolated private let processing = AtomicFlag()
    nonisolated private let droppedFrames = AtomicCounter()

    nonisolated private let instrumentation = FrameInstrumentation()

    // Stage 6, A/B path only: Kalman smoothing of the post-calibration screen
    // point. Used iff `PipelineTuning.smoother == .kalman`; the One Euro path
    // filters upstream of the projection instead and leaves this untouched.
    private let gazeKalman = GazeKalman()
    private var lastKalmanTime: CFTimeInterval?

    /// Speaks each word selected in Experiment 3. Held here rather than on the
    /// controller so the synthesizer (and its audio session) survives across
    /// runs — re-creating it per run drops the first word of each run while
    /// the session activates.
    private let wordAudio = WordAudioPlayer()

    var gazeEstimatorLoaded: Bool { pipeline.gazeEstimator.isModelLoaded }

    /// True when the active calibration has been validated and the verdict is
    /// good enough to hand to an experiment.
    ///
    /// The validation step existed but was advisory: experiments were gated on
    /// `calibration != nil` alone, so a `poor` verdict could be dismissed and
    /// the run would proceed. In the logged data that produced a 9×9 run with
    /// 0 % hits across all 324 trials and ~250 pt error at every eccentricity —
    /// a whole session spent measuring a broken fit. The gate is now binding.
    var isCalibrationUsable: Bool {
        guard calibration != nil else { return false }
        switch lastValidationVerdict {
        case .good, .marginal: return true
        case .poor, .incomplete, nil: return false
        }
    }

    /// Why the experiment triggers are unavailable, for the HUD. Nil when
    /// they're available.
    var calibrationBlockReason: String? {
        guard calibration != nil else { return "Calibrate first" }
        switch lastValidationVerdict {
        case .good, .marginal: return nil
        case nil: return "Validate calibration first"
        case .incomplete: return "Validation incomplete — re-validate"
        case .poor: return "Calibration poor — re-calibrate"
        }
    }

    /// True when no calibration / validation / experiment run is in progress
    /// and no results screen is up — i.e. the main screen owns the display.
    ///
    /// Both the live gaze dot and the trigger buttons are gated on this. It
    /// lives here rather than in the view because every new run or results
    /// screen has to be added to the condition, and having one copy means a
    /// new experiment can't be half-wired into it.
    var isIdle: Bool {
        calibrationController == nil
            && validationController == nil
            && validationResult == nil
            && accuracyController == nil
            && accuracyResult == nil
            && gridExperimentController == nil
            && gridExperimentResult == nil
            && fixationController == nil
            && fixationResult == nil
            && commTaskController == nil
            && commTaskResult == nil
    }

    // Camera buffers waiting for their landmarks. The cache is its own
    // (non-isolated) class so the camera queue and the main actor can both
    // touch it without fighting Swift concurrency. `nonisolated` because the
    // class is internally synchronized — Swift's actor checker doesn't know that.
    nonisolated private let pendingBuffers = PendingBufferCache()

    private var lastFrameTimes: [CFTimeInterval] = []

    init() {
        let svc = FaceLandmarkerService()
        landmarkerService = svc
        cameraManager.delegate = self
        svc?.delegate = self
    }

    func start() {
        cameraManager.requestAuthorization { [weak self] granted in
            guard let self = self, granted else {
                print("[Gaze] Camera permission denied.")
                return
            }
            self.cameraManager.configure()
            self.cameraManager.start()
        }
    }

    func stop() {
        cameraManager.stop()
        instrumentation.flush()
    }

    /// Begin a 9-dot calibration session. Requires `screenSize` to be set
    /// (driven by ContentView's GeometryReader). Discards the prior
    /// calibration so the gaze dot stops rendering during the run.
    func startCalibration() {
        guard screenSize.width > 0, screenSize.height > 0 else {
            print("[Gaze] Stage 5: cannot start calibration without screen size.")
            return
        }
        let targets = ScreenMapper.standardTargets(screenSize: screenSize)
        let c = CalibrationController(targets: targets)
        self.calibration = nil
        self.gazeScreenPoint = nil
        self.gazeKalman.reset()
        self.lastKalmanTime = nil
        self.resetSmoothers()
        self.validationResult = nil
        self.validationController = nil
        self.lastValidationVerdict = nil
        self.lastValidationMeanDegrees = .nan
        self.calibrationController = c
        c.start()
    }

    func cancelCalibration() {
        calibrationController?.cancel()
        calibrationController = nil
    }

    /// Begin a validation run against the current calibration. Called
    /// automatically when a calibration fit completes, and again from the
    /// results screen's "Re-validate" button.
    func startCalibrationValidation() {
        guard let cal = calibration else {
            print("[Gaze] Validation: calibrate first.")
            return
        }
        guard screenSize.width > 0, screenSize.height > 0 else {
            print("[Gaze] Validation: missing screen size.")
            return
        }
        let c = CalibrationValidationController(calibration: cal,
                                                screenSize: screenSize)
        c.onComplete = { [weak self] r in
            guard let self = self else { return }
            self.validationResult = r
            self.lastValidationVerdict = r.verdict
            self.lastValidationMeanDegrees = r.meanErrorDegrees
            self.validationController = nil
        }
        self.validationResult = nil
        self.validationController = c
        c.start()
    }

    func cancelCalibrationValidation() {
        validationController?.cancel()
        validationController = nil
    }

    /// Accept the validated calibration and dismiss the results screen.
    func acceptValidation() {
        validationResult = nil
    }

    /// Throw the calibration away and start a fresh one. Used by the
    /// validation results screen when the verdict is poor — this is the whole
    /// point of validating: a bad fit is caught here rather than silently
    /// corrupting an experiment.
    func discardCalibrationAndRecalibrate() {
        validationResult = nil
        validationController = nil
        startCalibration()
    }

    /// Begin a 5-point cross accuracy test. Requires a fitted calibration
    /// and a non-zero screen size. Clears any previous result.
    func startAccuracyTest() {
        guard let cal = calibration else {
            print("[Gaze] Accuracy test: calibrate first.")
            return
        }
        guard screenSize.width > 0, screenSize.height > 0 else {
            print("[Gaze] Accuracy test: missing screen size.")
            return
        }
        let targets = AccuracyTestController.crossTargets(screenSize: screenSize)
        let c = AccuracyTestController(targets: targets, calibration: cal)
        self.accuracyResult = nil
        self.accuracyController = c
        c.start()
    }

    func cancelAccuracyTest() {
        accuracyController?.cancel()
        accuracyController = nil
    }

    func dismissAccuracyResult() {
        accuracyResult = nil
    }

    /// Begin Experiment 1 (grid focus). Requires a fitted calibration and a
    /// non-zero screen size.
    func startGridExperiment(_ size: GridSize) {
        guard let cal = calibration else {
            print("[Gaze] Grid experiment: calibrate first.")
            return
        }
        guard screenSize.width > 0, screenSize.height > 0 else {
            print("[Gaze] Grid experiment: missing screen size.")
            return
        }
        let c = GridExperimentController(
            gridSize: size,
            screenSize: screenSize,
            calibration: cal
        )
        self.gridExperimentResult = nil
        self.gridExperimentController = c
        c.start()
    }

    func cancelGridExperiment() {
        gridExperimentController?.cancel()
        gridExperimentController = nil
    }

    func dismissGridExperimentResult() {
        gridExperimentResult = nil
    }

    /// Begin a fine-tune data-collection run. Drives the existing Experiment
    /// 1 controller forced to 5×4 mode, with a `FineTuneDataCollector`
    /// attached so each capture-window frame dumps a PNG + labels row.
    func startFineTuneCollection() {
        guard let cal = calibration else {
            print("[Gaze] Fine-tune collection: calibrate first.")
            return
        }
        guard screenSize.width > 0, screenSize.height > 0 else {
            print("[Gaze] Fine-tune collection: missing screen size.")
            return
        }
        guard let intr = cameraManager.intrinsics else {
            print("[Gaze] Fine-tune collection: no camera intrinsics yet.")
            return
        }
        let fiveByFour = GridSize.presets.first { $0.rows == 5 && $0.cols == 4 }
            ?? GridSize(rows: 5, cols: 4, repeats: 1, label: "5×4",
                        estimatedMinutes: 1.0, walking: false)
        let c = GridExperimentController(
            gridSize: fiveByFour,
            screenSize: screenSize,
            calibration: cal
        )
        c.dataCollector = FineTuneDataCollector(
            screenSize: screenSize,
            intrinsics: intr,
            calibration: cal
        )
        self.gridExperimentResult = nil
        self.gridExperimentController = c
        c.start()
    }

    /// Begin Experiment 2 (fixation stability). Requires a fitted calibration
    /// and a non-zero screen size.
    func startFixationExperiment() {
        guard let cal = calibration else {
            print("[Gaze] Experiment 2: calibrate first.")
            return
        }
        guard screenSize.width > 0, screenSize.height > 0 else {
            print("[Gaze] Experiment 2: missing screen size.")
            return
        }
        let c = FixationStabilityController(screenSize: screenSize,
                                            calibration: cal)
        c.onComplete = { [weak self] r in
            guard let self = self else { return }
            self.fixationResult = r
            self.fixationController = nil
            do {
                _ = try r.appendToMasterLog()
            } catch {
                print("experiment2 log append failed: \(error)")
            }
        }
        self.fixationResult = nil
        self.fixationController = c
        c.start()
    }

    func cancelFixationExperiment() {
        fixationController?.cancel()
        fixationController = nil
    }

    func dismissFixationResult() {
        fixationResult = nil
    }

    /// Begin Experiment 3 (image-based communication task). Requires a fitted
    /// calibration and a non-zero screen size.
    func startCommunicationTask() {
        guard let cal = calibration else {
            print("[Gaze] Experiment 3: calibrate first.")
            return
        }
        guard screenSize.width > 0, screenSize.height > 0 else {
            print("[Gaze] Experiment 3: missing screen size.")
            return
        }
        let c = CommunicationTaskController(screenSize: screenSize,
                                            calibration: cal,
                                            audio: wordAudio)
        c.onComplete = { [weak self] r in
            guard let self = self else { return }
            // Experiment 3 shows no results screen: the run ends straight
            // back to the idle view. The result is still logged and bundled
            // exactly as before — only the on-device presentation is gone,
            // so the data is read off the master log rather than the phone.
            self.lastCommTaskResult = r
            self.commTaskResult = nil
            self.commTaskController = nil
            do {
                _ = try r.appendToMasterLog()
            } catch {
                print("experiment3 log append failed: \(error)")
            }
        }
        self.lastCommTaskResult = nil
        self.commTaskResult = nil
        self.commTaskController = c
        c.start()
    }

    func cancelCommunicationTask() {
        commTaskController?.cancel()
        commTaskController = nil
    }

    func dismissCommunicationTaskResult() {
        commTaskResult = nil
    }

    /// Speak the sentence the participant composed. No longer reachable from
    /// a results screen (Experiment 3 has none); kept so the last run's
    /// sentence can still be replayed if a caller wants it.
    func replayComposedSentence() {
        guard let r = commTaskResult ?? lastCommTaskResult else { return }
        wordAudio.speakSentence(r.composedSentence)
    }

    /// Pull a fine-tuned `.mlpackage` from the laptop URL the user pasted in
    /// the fetch sheet, compile it, and hot-swap `gazeEstimator`'s internal
    /// model. Surfaces status to the UI via `modelFetchStatus`.
    ///
    /// The `.mlpackage` directory layout requires iOS 16+ for runtime
    /// compilation. On iOS 15 we report the version requirement and abort
    /// — there's no clean alternative without rebuilding the app.
    func fetchModel(baseURLString: String) {
        let s = baseURLString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !s.isEmpty,
              let url = URL(string: s.hasSuffix("/") ? s : (s + "/")) else {
            modelFetchStatus = "Invalid URL."
            return
        }
        modelFetchInFlight = true
        modelFetchStatus = "Fetching manifest…"
        Task {
            do {
                let pkgURL = try await ModelUpdater.fetch(baseURL: url) { p in
                    Task { @MainActor in
                        let mb = Double(p.bytesSoFar) / 1_048_576.0
                        self.modelFetchStatus = String(
                            format: "Downloading %d/%d  (%.2f MB)",
                            p.fileIndex, p.fileCount, mb)
                    }
                }
                guard #available(iOS 16.0, *) else {
                    self.modelFetchStatus = "iOS 16+ required to load .mlpackage at runtime."
                    self.modelFetchInFlight = false
                    return
                }

                // Pause the camera + MediaPipe while CoreML compiles the
                // new model. CoreML's compiler is memory-hungry, and the
                // live pipeline holds enough resident memory that the
                // combined peak frequently exceeds the iOS per-app limit
                // and gets the process jetsam-killed silently.
                self.modelFetchStatus = "Pausing camera, compiling…"
                self.cameraManager.stop()
                // Give the camera pipeline a moment to actually release.
                try? await Task.sleep(nanoseconds: 300_000_000)

                do {
                    let compiledURL = try await MLModel.compileModel(at: pkgURL)
                    // The swap happens on the pipeline queue even though the
                    // camera is stopped: a frame already dispatched there
                    // would otherwise be mid-`prediction` while the model is
                    // replaced under it.
                    let inputName = try self.processQueue.sync {
                        try self.pipeline.gazeEstimator.replace(
                            compiledModelAt: compiledURL,
                            sourceLabel: pkgURL.lastPathComponent
                        )
                    }
                    self.modelSource = pkgURL.lastPathComponent
                    self.modelFetchStatus = "Loaded \(pkgURL.lastPathComponent) (input=\(inputName))."
                    self.modelFetchSuccess = pkgURL.lastPathComponent
                } catch {
                    self.modelFetchStatus = "Compile failed: \(error.localizedDescription)"
                }
                // Always restart the camera, success or fail.
                self.cameraManager.start()
                self.modelFetchInFlight = false
            } catch {
                self.modelFetchStatus = "Fetch failed: \(error.localizedDescription)"
                self.modelFetchInFlight = false
            }
        }
    }

    /// Clear every smoother's history. Called on track loss and whenever a
    /// new calibration run starts, so a filter can't slew in from a state
    /// that belongs to a different fit.
    private func resetSmoothers() {
        gazeKalman.reset()
        lastKalmanTime = nil
        processQueue.async { self.pipeline.reset() }
    }

    private func recordFrameTime() {
        let now = CACurrentMediaTime()
        lastFrameTimes.append(now)
        while let first = lastFrameTimes.first, now - first > 1.0 {
            lastFrameTimes.removeFirst()
        }
        fps = Double(lastFrameTimes.count)
    }

}

extension GazeViewModel: CameraManagerDelegate {
    nonisolated func cameraManager(_ manager: CameraManager,
                                   didOutput sampleBuffer: CMSampleBuffer,
                                   orientation: UIImage.Orientation) {
        // Use the sample buffer's own presentation timestamp — it's strictly
        // monotonic from the AV pipeline, unlike CACurrentMediaTime() which
        // can collide when quantised to ms at high frame rates. MediaPipe's
        // live-stream API rejects non-increasing timestamps and stops
        // emitting landmarks entirely if any frame violates this.
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let timestampMs = Int(CMTimeGetSeconds(pts) * 1000)

        // Stash the image buffer keyed by timestamp so Stage 3 can pick it up
        // when the matching landmarks come back from MediaPipe.
        if let pb = CMSampleBufferGetImageBuffer(sampleBuffer) {
            pendingBuffers.store(pb, timestampMs: timestampMs)
        }

        Task { @MainActor in
            if self.imageSize == .zero, manager.imageWidth > 0 {
                self.imageSize = CGSize(width: manager.imageWidth,
                                        height: manager.imageHeight)
            }
        }
        // Call MediaPipe synchronously from the camera's serial queue so
        // frames stay in submission order — Task.detached would let the
        // runtime reorder them, also violating MediaPipe's monotonicity.
        landmarkerService?.detectAsync(
            sampleBuffer: sampleBuffer,
            orientation: orientation,
            timestampMs: timestampMs
        )
    }
}

extension GazeViewModel: FaceLandmarkerServiceDelegate {

    /// MediaPipe landmark callback. Runs on MediaPipe's own thread.
    ///
    /// Two things happen here and nothing else: backpressure, and a hand-off
    /// to `processQueue`. Everything the old implementation did inline —
    /// PnP, two warps, CoreML, projection — now runs off the main actor in
    /// `GazePipeline`.
    nonisolated func faceLandmarkerService(_ service: FaceLandmarkerService,
                                           didDetect result: FaceLandmarkerResult,
                                           timestampMs: Int) {
        let arrival = CACurrentMediaTime()

        // Backpressure. If a frame is still in flight, this one is *dropped*,
        // never queued: its buffer is drained out of the cache so it can't sit
        // there holding an IOSurface, and we return. Queuing instead would
        // trade a dropped frame for unbounded latency — and a gaze estimate
        // that arrives 200 ms late is worse than no estimate at all, because
        // it is indistinguishable from a current one in the log.
        guard processing.tryAcquire() else {
            _ = pendingBuffers.take(timestampMs: timestampMs)
            droppedFrames.increment()
            return
        }

        let faces = result.faceLandmarks
        let firstFace = faces.first ?? []
        let pb = pendingBuffers.take(timestampMs: timestampMs)
        let intr = cameraManager.intrinsics
        let dropped = droppedFrames.drain()

        processQueue.async { [self] in
            let out = pipeline.process(landmarks: firstFace,
                                       faceCount: faces.count,
                                       pixelBuffer: pb,
                                       intrinsics: intr,
                                       timestampMs: timestampMs,
                                       arrival: arrival,
                                       droppedSince: dropped)
            let workerDone = CACurrentMediaTime()
            Task { @MainActor in
                self.publish(out, workerDone: workerDone)
                // Released only after the publish completes, so the flag
                // measures the true end-to-end occupancy of the pipeline.
                self.processing.release()
            }
        }
    }

    nonisolated func faceLandmarkerService(_ service: FaceLandmarkerService,
                                           didFailWith error: Error) {
        print("[Gaze] Detection error: \(error)")
    }
}

// MARK: - Main-actor publish

extension GazeViewModel {

    /// The only part of the per-frame path still on the main actor:
    /// assign the `@Published` properties, project the (already-filtered)
    /// gaze through the calibration, and feed the experiment controllers —
    /// all of which are main-actor `ObservableObject`s driving the UI.
    ///
    /// The projection itself is two divides and a nearest-target scan; it
    /// stays here so the controllers see a point computed from exactly the
    /// calibration that is live at publish time.
    fileprivate func publish(_ out: GazePipeline.Output,
                             workerDone: CFTimeInterval) {
        var timing = out.timing

        faceCount = out.faceCount
        landmarks = out.landmarks
        headPose = out.poseFiltered
        headPoseFailureReason = out.poseFailureReason
        if let s = out.intrinsicsSummary { intrinsicsSummary = s }
        if imageSize == .zero, cameraManager.imageWidth > 0 {
            imageSize = CGSize(width: cameraManager.imageWidth,
                               height: cameraManager.imageHeight)
        }

        // Debug crops: throttled to every Nth frame. The eye strip was not
        // even computed on the other frames (see `GazePipeline`); the face
        // crop was, because Experiment 1's fine-tune collector consumes it.
        if out.publishDebugCrops {
            if let strip = out.eyeStrip { normalizedEyes = strip }
            if let tensor = out.eyeTensor { normalizedEyesTensor = tensor }
            normalizedFace = out.faceImage
        }

        gazeOriginCam = out.eyeMidFiltered ?? out.eyeMidRaw
        gaze = out.filteredEstimate

        guard let filtered = out.filteredEstimate else {
            // No usable estimate this frame — drop the dot rather than leave
            // a stale one on screen asserting a gaze we do not have.
            gazeScreenPoint = nil
            gazeKalman.reset()
            lastKalmanTime = nil
            recordFrameTime()
            finishTiming(&timing, workerDone: workerDone)
            return
        }

        let toDeg = 180.0 / Double.pi
        let halfW = Double(screenSize.width) * 0.5
        let halfH = Double(screenSize.height) * 0.5
        let haveScreen = screenSize.width > 0

        // ---- Two streams, logged separately ------------------------------
        //
        // `raw*` is populated only when the CNN actually ran this frame. On a
        // blink-gated frame every raw field stays NaN rather than repeating
        // the held value: a duplicate in the raw stream would understate its
        // variance, and Experiment 2's whole job is to measure that variance.
        var diag = ExperimentRunLog.Diagnostics()
        diag.ear = out.ear.mean
        diag.blinkHeld = out.blinkHeld
        diag.eyeCam = out.eyeMidRaw ?? simd_double3(.nan, .nan, .nan)
        diag.filtPitchDeg = filtered.pitch * toDeg
        diag.filtYawDeg = filtered.yaw * toDeg

        var rawScreenPoint: CGPoint?
        if let raw = out.rawEstimate {
            diag.rawPitchDeg = raw.pitch * toDeg
            diag.rawYawDeg = raw.yaw * toDeg
            if let cal = calibration, haveScreen {
                let p = cal.predict(gazeCam: raw.gazeCam)
                if p.x.isFinite, p.y.isFinite {
                    diag.rawPredX = halfW + p.x
                    diag.rawPredY = halfH + p.y
                    rawScreenPoint = CGPoint(x: diag.rawPredX, y: diag.rawPredY)
                }
            }
        }

        // Iris-diameter proxy for the run logs. Measured only while an
        // experiment is recording — it's an extra landmark pass on the gaze
        // path otherwise wasted.
        let pupil: PupilMeasure.Diameters
        if let intr = cameraManager.intrinsics,
           gridExperimentController != nil || fixationController != nil
            || commTaskController != nil || validationController != nil {
            pupil = PupilMeasure.diameters(
                landmarks: out.landmarks,
                imageSize: CGSize(width: intr.imageWidth,
                                  height: intr.imageHeight))
        } else {
            pupil = .init(leftPx: .nan, rightPx: .nan)
        }

        // ---- Calibration / Experiment 1 / accuracy: raw stream -----------
        //
        // These fit or score on the unsmoothed estimate, exactly as before
        // this branch: `CalibrationController` takes per-dot medians, which
        // is its own (and better) noise rejection, and pre-smoothing its
        // input would bias the fit toward whatever the filter was doing.
        if let raw = out.rawEstimate, let pose = out.poseFiltered {
            if let c = calibrationController {
                c.ingest(gazeCam: raw.gazeCam)
                if c.phase == .complete, let r = c.result {
                    calibration = r
                    calibrationController = nil
                    resetSmoothers()
                    // Validation follows calibration automatically: the fit is
                    // never handed to an experiment until it has been checked
                    // against all 9 dots.
                    startCalibrationValidation()
                }
            }
            if let a = accuracyController {
                a.ingest(gazeCam: raw.gazeCam, headPose: pose)
                if a.phase == .complete, let r = a.result {
                    accuracyResult = r
                    accuracyController = nil
                    do { _ = try r.appendToMasterLog() }
                    catch { print("accuracy log append failed: \(error)") }
                }
            }
            if let ge = gridExperimentController {
                ge.ingest(gazeCam: raw.gazeCam,
                          headPose: pose,
                          faceImage: out.faceImage,
                          normalizationRotation: out.faceRotation,
                          eyePositionCam: out.eyeMidRaw ?? simd_double3(),
                          pupilDiameters: pupil,
                          diagnostics: diag)
                if ge.phase == .complete, let r = ge.result {
                    gridExperimentResult = r
                    gridExperimentController = nil
                    do { _ = try r.appendToMasterLog() }
                    catch { print("grid experiment log append failed: \(error)") }
                }
            }
        }

        // ---- Projection of the filtered stream ---------------------------
        guard let cal = calibration, haveScreen else {
            gazeScreenPoint = nil
            recordFrameTime()
            finishTiming(&timing, workerDone: workerDone, diag: diag)
            return
        }
        let p = cal.predict(gazeCam: filtered.gazeCam)
        guard p.x.isFinite, p.y.isFinite else {
            recordFrameTime()
            finishTiming(&timing, workerDone: workerDone, diag: diag)
            return
        }
        let projected = CGPoint(x: halfW + p.x, y: halfH + p.y)

        // In the One Euro configuration the smoothing already happened
        // upstream, on (pitch, yaw), so this point is final. In the Kalman
        // A/B configuration the pipeline passed the angles through untouched
        // and the old post-projection filter runs here instead.
        let rendered: CGPoint
        switch PipelineTuning.smoother {
        case .oneEuro:
            rendered = projected
        case .kalman:
            let now = CACurrentMediaTime()
            let dt = lastKalmanTime.map { now - $0 } ?? 0
            lastKalmanTime = now
            rendered = gazeKalman.update(measurement: projected, dt: dt)
        }
        gazeScreenPoint = rendered
        diag.filtPredX = Double(rendered.x)
        diag.filtPredY = Double(rendered.y)

        guard let pose = out.poseFiltered else {
            recordFrameTime()
            finishTiming(&timing, workerDone: workerDone, diag: diag)
            return
        }

        // ---- Experiment 2 --------------------------------------------------
        //
        // Which stream is scored is `PipelineTuning.fixationScoredStream`,
        // and it changes what the result means:
        //
        //   .raw       the per-frame CNN output, no temporal smoothing and no
        //              blink hold — the estimator's noise floor. Blink-gated
        //              frames deliver nothing, so a held-over estimate can
        //              never enter the scatter as a zero-variance sample.
        //   .filtered  the same point the cursor is drawn from, post One Euro
        //              (or Kalman). Reports what the user actually sees and
        //              acts on; will always look tighter than .raw, because
        //              shrinking that variance is the smoother's whole job.
        //
        // Either way the *other* stream still rides along in `diag`, so an
        // off-device analysis can compare the two after the fact.
        switch PipelineTuning.fixationScoredStream {
        case .raw:
            if let rawPoint = rawScreenPoint, let raw = out.rawEstimate {
                fixationController?.ingest(predictionScreenAbs: rawPoint,
                                           gazeCam: raw.gazeCam,
                                           headPose: pose,
                                           pupilDiameters: pupil,
                                           diagnostics: diag)
            }
        case .filtered:
            fixationController?.ingest(predictionScreenAbs: rendered,
                                       gazeCam: filtered.gazeCam,
                                       headPose: pose,
                                       pupilDiameters: pupil,
                                       diagnostics: diag)
        }

        // ---- Validation and Experiment 3: filtered stream ----------------
        //
        // Both of these are interaction tasks scored on dwell: what matters
        // is whether the participant could land the dot they can see, so they
        // consume the same point that is rendered.
        validationController?.ingest(predictionScreenAbs: rendered,
                                     gazeCam: filtered.gazeCam,
                                     headPose: pose,
                                     pupilDiameters: pupil,
                                     diagnostics: diag)
        commTaskController?.ingest(predictionScreenAbs: rendered,
                                   gazeCam: filtered.gazeCam,
                                   headPose: pose,
                                   pupilDiameters: pupil,
                                   diagnostics: diag)

        recordFrameTime()
        finishTiming(&timing, workerDone: workerDone, diag: diag)
    }

    /// Stamp the publish-side timings and hand the row to the logger.
    private func finishTiming(_ timing: inout FrameInstrumentation.Row,
                              workerDone: CFTimeInterval,
                              diag: ExperimentRunLog.Diagnostics? = nil) {
        let now = CACurrentMediaTime()
        timing.publish = now
        timing.dtPublish = now - workerDone
        if let d = diag {
            timing.rawPredX = d.rawPredX
            timing.rawPredY = d.rawPredY
            timing.filtPredX = d.filtPredX
            timing.filtPredY = d.filtPredY
        }
        instrumentation.record(timing)
    }
}
