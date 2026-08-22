import Foundation
import AVFoundation
import CoreML
import MediaPipeTasksVision
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
    @Published var normalizedEyesTensor: [Float] = []
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

    let cameraManager = CameraManager()
    // Nonisolated so the camera queue can call detectAsync synchronously
    // without an actor hop (which would let Swift reorder frame submission
    // and trip MediaPipe's monotonic-timestamp requirement).
    nonisolated private let landmarkerService: FaceLandmarkerService?
    private let headPoseEstimator = HeadPoseEstimator()
    private let eyeNormalizer = EyeNormalizer()
    private let faceNormalizer = FaceNormalizer()
    private let gazeEstimator = GazeEstimator()

    // Stage 6: Kalman smoothing of the post-calibration screen point.
    private let gazeKalman = GazeKalman()
    private var lastKalmanTime: CFTimeInterval?

    /// Speaks each word selected in Experiment 3. Held here rather than on the
    /// controller so the synthesizer (and its audio session) survives across
    /// runs — re-creating it per run drops the first word of each run while
    /// the session activates.
    private let wordAudio = WordAudioPlayer()

    var gazeEstimatorLoaded: Bool { gazeEstimator.isModelLoaded }

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
            self.commTaskResult = r
            self.commTaskController = nil
            do {
                _ = try r.appendToMasterLog()
            } catch {
                print("experiment3 log append failed: \(error)")
            }
        }
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

    /// Speak the sentence the participant composed, from the results screen.
    func replayComposedSentence() {
        guard let r = commTaskResult else { return }
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
                    let inputName = try self.gazeEstimator.replace(
                        compiledModelAt: compiledURL,
                        sourceLabel: pkgURL.lastPathComponent
                    )
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
    nonisolated func faceLandmarkerService(_ service: FaceLandmarkerService,
                                           didDetect result: FaceLandmarkerResult,
                                           timestampMs: Int) {
        let faces = result.faceLandmarks
        let firstFace = faces.first ?? []
        Task { @MainActor in
            self.faceCount = faces.count
            self.landmarks = firstFace

            // Stage 2: head pose.
            var pose: HeadPose?
            if firstFace.isEmpty {
                self.headPoseFailureReason = "no face"
            } else if let intr = self.cameraManager.intrinsics {
                if self.intrinsicsSummary == nil {
                    self.intrinsicsSummary = String(
                        format: "fx=%.0f cx=%.0f img=%.0fx%.0f",
                        intr.fx, intr.cx, intr.imageWidth, intr.imageHeight
                    )
                }
                pose = self.headPoseEstimator.estimate(
                    landmarks: firstFace, intrinsics: intr
                )
                self.headPoseFailureReason = (pose == nil)
                    ? self.headPoseEstimator.lastFailureReason
                    : nil
            } else {
                self.headPoseFailureReason = "no intrinsics"
            }
            self.headPose = pose

            // Stage 3: head-pose-normalized eye strip. Always drain the buffer
            // for this timestamp so it doesn't sit in the cache; only run the
            // warp if we have a valid pose + intrinsics.
            let pb = self.pendingBuffers.take(timestampMs: timestampMs)
            if let pose = pose,
               let intr = self.cameraManager.intrinsics,
               let pb = pb,
               let out = self.eyeNormalizer.normalize(
                   pixelBuffer: pb,
                   headPose: pose,
                   intrinsics: intr
               ) {
                self.normalizedEyes = out.combined
                self.normalizedEyesTensor = out.tensorRGB

                // Stage 4: ETH-XGaze ResNet18 takes a 224×224 normalized
                // FACE crop. Run that warp in parallel with the eye strip
                // so we keep the eye-strip viz from Phase 3.
                let eL = pose.rotation * CanonicalFaceModel.leftEyeCenter3D  + pose.translation
                let eR = pose.rotation * CanonicalFaceModel.rightEyeCenter3D + pose.translation
                let eyeMid = (eL + eR) * 0.5
                self.gazeOriginCam = eyeMid
                let face = self.faceNormalizer.normalize(
                    pixelBuffer: pb,
                    headPose: pose,
                    intrinsics: intr
                )
                if let face = face {
                    self.normalizedFace = face.image
                    self.gaze = self.gazeEstimator.estimate(
                        tensorRGB: face.tensorRGB,
                        normalizationRotation: face.rotation
                    )
                } else {
                    self.normalizedFace = nil
                    self.gaze = nil
                }

                // Stage 5: feed calibration and/or project to screen coords.
                if let g = self.gaze {
                    // Per-frame diagnostics for the experiment run logs: the
                    // same frame at raw / projected stages so noise can be
                    // attributed off-device. This build has no upstream
                    // smoother, so the `filt_*` / `eye_cam_*_f_mm` columns
                    // stay empty by design.
                    let toDeg = 180.0 / Double.pi
                    var diag = ExperimentRunLog.Diagnostics()
                    diag.rawPitchDeg = g.pitch * toDeg
                    diag.rawYawDeg = g.yaw * toDeg
                    diag.eyeCam = eyeMid
                    if let cal = self.calibration, self.screenSize.width > 0 {
                        let pRaw = cal.predict(gazeCam: g.gazeCam)
                        diag.rawPredX = Double(self.screenSize.width) * 0.5 + pRaw.x
                        diag.rawPredY = Double(self.screenSize.height) * 0.5 + pRaw.y
                    }

                    // Iris-diameter proxy for the run logs. Measured only
                    // while an experiment is recording — it's an extra
                    // landmark pass on the gaze path otherwise wasted.
                    let pupil: PupilMeasure.Diameters
                    if self.gridExperimentController != nil
                        || self.fixationController != nil
                        || self.commTaskController != nil
                        || self.validationController != nil {
                        pupil = PupilMeasure.diameters(
                            landmarks: firstFace,
                            imageSize: CGSize(width: intr.imageWidth,
                                              height: intr.imageHeight))
                    } else {
                        pupil = .init(leftPx: .nan, rightPx: .nan)
                    }

                    if let c = self.calibrationController {
                        c.ingest(gazeCam: g.gazeCam)
                        if c.phase == .complete, let r = c.result {
                            self.calibration = r
                            self.calibrationController = nil
                            // Validation follows calibration automatically:
                            // the fit is never handed to an experiment until
                            // it has been checked against all 9 dots.
                            self.startCalibrationValidation()
                        }
                    }
                    if let a = self.accuracyController {
                        a.ingest(gazeCam: g.gazeCam, headPose: pose)
                        if a.phase == .complete, let r = a.result {
                            self.accuracyResult = r
                            self.accuracyController = nil
                            // Cumulative log: append this run's section to
                            // accuracy_log.csv so every run is preserved
                            // even if the user doesn't tap Export CSV.
                            do {
                                _ = try r.appendToMasterLog()
                            } catch {
                                print("accuracy log append failed: \(error)")
                            }
                        }
                    }
                    if let ge = self.gridExperimentController {
                        ge.ingest(gazeCam: g.gazeCam,
                                  headPose: pose,
                                  faceImage: face?.image,
                                  normalizationRotation: face?.rotation,
                                  eyePositionCam: eyeMid,
                                  pupilDiameters: pupil,
                                  diagnostics: diag)
                        if ge.phase == .complete, let r = ge.result {
                            self.gridExperimentResult = r
                            self.gridExperimentController = nil
                            do {
                                _ = try r.appendToMasterLog()
                            } catch {
                                print("grid experiment log append failed: \(error)")
                            }
                        }
                    }
                    if let cal = self.calibration, self.screenSize.width > 0 {
                        let p = cal.predict(gazeCam: g.gazeCam)
                        if p.x.isFinite && p.y.isFinite {
                            let raw = CGPoint(
                                x: self.screenSize.width  * 0.5 + p.x,
                                y: self.screenSize.height * 0.5 + p.y
                            )
                            // Stage 6: Kalman smoothing with real per-frame dt.
                            let now = CACurrentMediaTime()
                            let dt = self.lastKalmanTime.map { now - $0 } ?? 0
                            self.lastKalmanTime = now
                            let smoothed =
                                self.gazeKalman.update(measurement: raw, dt: dt)
                            self.gazeScreenPoint = smoothed

                            // Validation and Experiments 2/3 all consume the
                            // same Kalman-smoothed, screen-absolute point
                            // that's drawn on screen, so what the participant
                            // sees and what gets scored are identical.
                            //
                            // None of these harvest their result here: each
                            // finishes off the gaze path (timer or button) and
                            // reports through its `onComplete` callback, set
                            // when the run was started.
                            self.validationController?.ingest(
                                predictionScreenAbs: smoothed,
                                gazeCam: g.gazeCam,
                                headPose: pose,
                                pupilDiameters: pupil,
                                diagnostics: diag)
                            self.fixationController?.ingest(
                                predictionScreenAbs: smoothed,
                                gazeCam: g.gazeCam,
                                headPose: pose,
                                pupilDiameters: pupil,
                                diagnostics: diag)
                            self.commTaskController?.ingest(
                                predictionScreenAbs: smoothed,
                                gazeCam: g.gazeCam,
                                headPose: pose,
                                pupilDiameters: pupil,
                                diagnostics: diag)
                        }
                    }
                }
            } else {
                self.gaze = nil
                self.gazeOriginCam = nil
                self.normalizedFace = nil
                // Track lost — reset filter so re-acquisition doesn't snap from stale state.
                self.gazeKalman.reset()
                self.lastKalmanTime = nil
            }

            self.recordFrameTime()
        }
    }

    nonisolated func faceLandmarkerService(_ service: FaceLandmarkerService,
                                           didFailWith error: Error) {
        print("[Gaze] Detection error: \(error)")
    }
}
