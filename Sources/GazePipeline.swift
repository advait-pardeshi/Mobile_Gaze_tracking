import Foundation
import CoreVideo
import UIKit
import simd
import QuartzCore
import MediaPipeTasksVision

/// Stages 2–4 of the gaze pipeline, off the main actor.
///
/// Everything here — PnP, the two normalization warps, the CoreML call and the
/// smoothing filters — used to run inside a `Task { @MainActor }` hung off the
/// MediaPipe callback. At 30–60 Hz that is tens of milliseconds of numerical
/// work per frame on the thread that also drives SwiftUI, which is both the
/// source of the UI hitching and a reason the tail latency was unbounded:
/// `Task { @MainActor }` *queues*, so a slow frame pushed the next one behind
/// it instead of being dropped.
///
/// This type owns all the per-frame mutable state (filters, warm starts, the
/// blink hold) and is confined to one serial `DispatchQueue` by
/// `GazeViewModel`. It is deliberately **not** thread-safe on its own: the
/// single-queue confinement is the invariant, and adding locks here would
/// only hide a violation of it.
///
/// The main actor gets back an immutable `Output` and does three things with
/// it: publish, project through the calibration, and feed the experiment
/// controllers.
final class GazePipeline {

    // MARK: - Output

    struct Output {
        let timestampMs: Int
        let arrival: CFTimeInterval

        let faceCount: Int
        let landmarks: [NormalizedLandmark]

        /// Head pose straight out of PnP.
        let poseRaw: HeadPose?
        /// Head pose after the quaternion One Euro filter — this is the pose
        /// the normalization crop was actually taken with, so it is the one
        /// that gets published and logged as *the* head pose for this frame.
        let poseFiltered: HeadPose?
        let poseFailureReason: String?
        let intrinsicsSummary: String?

        /// 200×50 eye strip. Non-nil only on frames where the debug crops are
        /// due to be published — the strip feeds no downstream stage, so it is
        /// skipped entirely on the others rather than computed and discarded.
        let eyeStrip: UIImage?
        /// Companion tensor to `eyeStrip`, same throttle.
        let eyeTensor: [Float]?

        /// 224×224 normalized face crop. Computed every frame (Experiment 1's
        /// fine-tune collector consumes it); `publishDebugCrops` says whether
        /// it should also be assigned to the `@Published` property.
        let faceImage: UIImage?
        let faceRotation: simd_double3x3?
        let publishDebugCrops: Bool

        /// Eye midpoint in camera coords (mm), from the raw and the filtered
        /// head pose respectively.
        let eyeMidRaw: simd_double3?
        let eyeMidFiltered: simd_double3?

        /// This frame's CNN output. **Nil** when the CNN did not run — no
        /// face, no crop, or a blink-gated frame. Never a held-over value:
        /// the raw stream must contain only real measurements.
        let rawEstimate: GazeEstimator.Estimate?
        /// The smoothed estimate that drives the rendered dot. On a
        /// blink-gated frame this is the previous frame's filtered value,
        /// held.
        let filteredEstimate: GazeEstimator.Estimate?

        let ear: EyeAspectRatio.Ratios
        /// True iff the CNN was skipped for this frame because of the blink
        /// gate and `filteredEstimate` is therefore a hold.
        let blinkHeld: Bool

        /// Partially-filled instrumentation row; the main actor completes the
        /// `publish`/`dtPublish` fields and hands it to the logger.
        var timing: FrameInstrumentation.Row
    }

    // MARK: - State (serial-queue confined)

    private let headPoseEstimator = HeadPoseEstimator()
    private let eyeNormalizer = EyeNormalizer()
    private let faceNormalizer = FaceNormalizer()
    let gazeEstimator = GazeEstimator()

    private var headPoseFilter = OneEuroQuaternion(
        minCutoff: PipelineTuning.headPoseMinCutoff,
        beta: PipelineTuning.headPoseBeta,
        dCutoff: PipelineTuning.headPoseDCutoff)
    private var pitchFilter = OneEuroFilter(
        minCutoff: PipelineTuning.gazeAngleMinCutoff,
        beta: PipelineTuning.gazeAngleBeta,
        dCutoff: PipelineTuning.gazeAngleDCutoff)
    private var yawFilter = OneEuroFilter(
        minCutoff: PipelineTuning.gazeAngleMinCutoff,
        beta: PipelineTuning.gazeAngleBeta,
        dCutoff: PipelineTuning.gazeAngleDCutoff)
    private var eyePositionFilter = OneEuroVector3(
        minCutoff: PipelineTuning.eyePositionMinCutoff,
        beta: PipelineTuning.eyePositionBeta,
        dCutoff: PipelineTuning.eyePositionDCutoff)

    /// Adaptive blink gate. Replaces the fixed EAR threshold, which could
    /// not tell a blink from a downward gaze and so froze the dot whenever
    /// the participant looked at the bottom of the screen.
    private var blinkDetector = BlinkDetector()

    private var lastFilterTime: CFTimeInterval?
    /// Last filtered estimate, held across blink-gated frames.
    private var heldEstimate: GazeEstimator.Estimate?
    private var blinkHeldFrames = 0
    private var frameCounter = 0
    private var emittedIntrinsicsSummary = false

    /// Drop all filter history. Called on track loss and when calibration
    /// restarts, so re-acquisition doesn't slew in from a stale state.
    func reset() {
        headPoseFilter.reset()
        pitchFilter.reset()
        yawFilter.reset()
        eyePositionFilter.reset()
        blinkDetector.reset()
        lastFilterTime = nil
        heldEstimate = nil
        blinkHeldFrames = 0
    }

    // MARK: - Per-frame

    func process(landmarks: [NormalizedLandmark],
                 faceCount: Int,
                 pixelBuffer: CVPixelBuffer?,
                 intrinsics: CameraIntrinsics?,
                 timestampMs: Int,
                 arrival: CFTimeInterval,
                 droppedSince: Int) -> Output {

        frameCounter &+= 1
        let publishCrops =
            frameCounter % PipelineTuning.debugCropPublishInterval == 0

        var timing = FrameInstrumentation.Row()
        timing.timestampMs = timestampMs
        timing.arrival = arrival
        timing.droppedSince = droppedSince
        timing.dtDispatch = CACurrentMediaTime() - arrival

        // ---- Stage 2: head pose ------------------------------------------
        var pose: HeadPose?
        var failureReason: String?
        let t0 = CACurrentMediaTime()
        if landmarks.isEmpty {
            failureReason = "no face"
        } else if let intr = intrinsics {
            pose = headPoseEstimator.estimate(landmarks: landmarks,
                                              intrinsics: intr)
            failureReason = (pose == nil) ? headPoseEstimator.lastFailureReason : nil
        } else {
            failureReason = "no intrinsics"
        }
        timing.dtPose = CACurrentMediaTime() - t0
        timing.poseOK = (pose != nil)

        var summary: String?
        if let intr = intrinsics, !emittedIntrinsicsSummary {
            emittedIntrinsicsSummary = true
            summary = String(format: "fx=%.0f cx=%.0f img=%.0fx%.0f",
                             intr.fx, intr.cx, intr.imageWidth, intr.imageHeight)
        }

        // ---- Eye aspect ratio (always, even when the pose failed) ---------
        let ear: EyeAspectRatio.Ratios
        if let intr = intrinsics, !landmarks.isEmpty {
            ear = EyeAspectRatio.compute(
                landmarks: landmarks,
                imageSize: CGSize(width: intr.imageWidth, height: intr.imageHeight))
        } else {
            ear = EyeAspectRatio.Ratios(left: .nan, right: .nan)
        }
        timing.earLeft = ear.left
        timing.earRight = ear.right

        guard let rawPose = pose, let intr = intrinsics, let pb = pixelBuffer else {
            // Track lost: nothing downstream can be trusted, so clear the
            // filters rather than let them decay toward a stale estimate.
            reset()
            return Output(timestampMs: timestampMs, arrival: arrival,
                          faceCount: faceCount, landmarks: landmarks,
                          poseRaw: pose, poseFiltered: pose,
                          poseFailureReason: failureReason,
                          intrinsicsSummary: summary,
                          eyeStrip: nil, eyeTensor: nil,
                          faceImage: nil, faceRotation: nil,
                          publishDebugCrops: publishCrops,
                          eyeMidRaw: nil, eyeMidFiltered: nil,
                          rawEstimate: nil, filteredEstimate: nil,
                          ear: ear, blinkHeld: false, timing: timing)
        }

        let now = CACurrentMediaTime()
        let dt = lastFilterTime.map { now - $0 } ?? (1.0 / 30.0)
        lastFilterTime = now

        // ---- Stage 2b: smooth R_h before it reaches the crop -------------
        //
        // The 224×224 normalized face is a function of R_h, so PnP jitter
        // becomes a shimmering crop, and the CNN sees a slightly different
        // image of a perfectly still face on every frame. Filtering the pose
        // here attacks that noise at its source; filtering the CNN's output
        // afterwards can only average it down.
        let tHead0 = CACurrentMediaTime()
        let qRaw = rawPose.quaternion
        let qFiltered = headPoseFilter.update(qRaw, dt: dt)
        let posePublished = rawPose.withRotation(qFiltered)
        timing.dtHeadFilter = CACurrentMediaTime() - tHead0
        timing.qRaw = qRaw
        timing.qFiltered = qFiltered

        let eyeMidRaw = Self.eyeMidpoint(rawPose)
        timing.eyeRaw = eyeMidRaw

        // ---- Stage 3: debug eye strip (throttled) ------------------------
        var eyeStrip: UIImage?
        var eyeTensor: [Float]?
        if publishCrops {
            let tEye0 = CACurrentMediaTime()
            if let out = eyeNormalizer.normalize(pixelBuffer: pb,
                                                 headPose: posePublished,
                                                 intrinsics: intr) {
                eyeStrip = out.combined
                eyeTensor = out.tensorRGB
            }
            timing.dtEyeWarp = CACurrentMediaTime() - tEye0
        }

        // ---- Stage 3b: 224×224 face crop ---------------------------------
        let tFace0 = CACurrentMediaTime()
        let face = faceNormalizer.normalize(pixelBuffer: pb,
                                            headPose: posePublished,
                                            intrinsics: intr)
        timing.dtFaceWarp = CACurrentMediaTime() - tFace0
        timing.faceCropOK = (face != nil)

        guard let face = face else {
            reset()
            return Output(timestampMs: timestampMs, arrival: arrival,
                          faceCount: faceCount, landmarks: landmarks,
                          poseRaw: rawPose, poseFiltered: posePublished,
                          poseFailureReason: failureReason,
                          intrinsicsSummary: summary,
                          eyeStrip: eyeStrip, eyeTensor: eyeTensor,
                          faceImage: nil, faceRotation: nil,
                          publishDebugCrops: publishCrops,
                          eyeMidRaw: eyeMidRaw, eyeMidFiltered: nil,
                          rawEstimate: nil, filteredEstimate: nil,
                          ear: ear, blinkHeld: false, timing: timing)
        }

        // ---- Blink gate --------------------------------------------------
        //
        // A closed eye is out-of-distribution for ETH-XGaze, not merely hard:
        // the network still returns a confident (pitch, yaw) with nothing
        // behind it. Skipping the CNN outright is both cheaper and more
        // honest than trying to reject the answer afterwards.
        let meanEAR = ear.mean
        let isBlink = blinkDetector.isBlink(meanEAR: meanEAR)

        var rawEstimate: GazeEstimator.Estimate?
        var blinkHeld = false

        if isBlink {
            blinkHeldFrames += 1
            if blinkHeldFrames > PipelineTuning.blinkMaxHeldFrames {
                // Sustained closure is not a blink. Stop asserting a gaze.
                heldEstimate = nil
            }
            blinkHeld = heldEstimate != nil
        } else {
            blinkHeldFrames = 0
            let tCNN0 = CACurrentMediaTime()
            rawEstimate = gazeEstimator.estimate(tensorRGB: face.tensorRGB,
                                                 normalizationRotation: face.rotation)
            timing.dtCNN = CACurrentMediaTime() - tCNN0
        }
        timing.blinkHeld = blinkHeld
        timing.earBaseline = blinkDetector.currentBaseline

        // ---- Stage 4b: smoothing, BEFORE projection ----------------------
        let tFilt0 = CACurrentMediaTime()
        var eyeMidFiltered: simd_double3?
        var filtered: GazeEstimator.Estimate?

        if let raw = rawEstimate {
            timing.rawPitchDeg = raw.pitch * 180.0 / .pi
            timing.rawYawDeg = raw.yaw * 180.0 / .pi

            switch PipelineTuning.smoother {
            case .oneEuro:
                let p = pitchFilter.update(raw.pitch, dt: dt)
                let y = yawFilter.update(raw.yaw, dt: dt)
                filtered = GazeEstimator.estimate(
                    pitch: p, yaw: y, normalizationRotation: face.rotation)
            case .kalman:
                // A/B path: leave the angles untouched here and let
                // `GazeViewModel`'s `GazeKalman` smooth the projected point,
                // exactly as the previous build did.
                filtered = raw
            }
            heldEstimate = filtered
        } else {
            filtered = heldEstimate
        }

        // The eye-position filter runs every frame the pose is valid,
        // including blink-gated ones — the head keeps moving with the eyes
        // shut, and this filter has no dependence on the CNN.
        eyeMidFiltered = eyePositionFilter.update(eyeMidRaw, dt: dt)
        timing.eyeFiltered = eyeMidFiltered
        timing.dtGazeFilter = CACurrentMediaTime() - tFilt0

        if let f = filtered {
            timing.filtPitchDeg = f.pitch * 180.0 / .pi
            timing.filtYawDeg = f.yaw * 180.0 / .pi
        }

        return Output(timestampMs: timestampMs, arrival: arrival,
                      faceCount: faceCount, landmarks: landmarks,
                      poseRaw: rawPose, poseFiltered: posePublished,
                      poseFailureReason: failureReason,
                      intrinsicsSummary: summary,
                      eyeStrip: eyeStrip, eyeTensor: eyeTensor,
                      faceImage: face.image, faceRotation: face.rotation,
                      publishDebugCrops: publishCrops,
                      eyeMidRaw: eyeMidRaw, eyeMidFiltered: eyeMidFiltered,
                      rawEstimate: rawEstimate, filteredEstimate: filtered,
                      ear: ear, blinkHeld: blinkHeld, timing: timing)
    }

    /// Midpoint between the two eye centres in camera coords (mm) — the
    /// anchor for the gaze-ray overlay and the `eye_cam_*` log columns.
    private static func eyeMidpoint(_ pose: HeadPose) -> simd_double3 {
        let eL = pose.rotation * CanonicalFaceModel.leftEyeCenter3D + pose.translation
        let eR = pose.rotation * CanonicalFaceModel.rightEyeCenter3D + pose.translation
        return (eL + eR) * 0.5
    }
}
