import Foundation
import UIKit
import simd

/// Per-run data sink for the off-device fine-tune flow.
///
/// During a 5×4 grid-experiment run, for every capture-window frame we
/// dump the 224×224 face crop that the CNN saw plus the metadata the
/// laptop needs to derive (pitch, yaw) labels: the cell the user was
/// looking at, the normalization rotation `R_n`, the eye position in
/// camera coords (from PnP), and the current head pose.
///
/// A small `meta.json` and `calibration.json` are written once at
/// init time. On `finalize()` the entire run directory is zipped via
/// `NSFileCoordinator(.forUploading)` and the resulting URL is handed
/// back to the results view to surface through the share sheet.
@MainActor
final class FineTuneDataCollector {

    /// Directory holding `frames/`, `labels.csv`, `meta.json`,
    /// `calibration.json`.
    let runDirectory: URL
    /// Stable timestamped run name (also the zip's basename).
    let runName: String

    private let labelsURL: URL
    private let framesDir: URL
    private var labelsHandle: FileHandle?
    private var frameCount: Int = 0

    init?(screenSize: CGSize,
          intrinsics: CameraIntrinsics,
          calibration: CalibrationModel) {
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd_HHmmss"
        let stamp = df.string(from: Date())
        self.runName = "finetune_\(stamp)"

        let docs = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
        let parent = docs.appendingPathComponent("finetune_runs", isDirectory: true)
        let runDir = parent.appendingPathComponent(runName, isDirectory: true)
        let frames = runDir.appendingPathComponent("frames", isDirectory: true)

        do {
            try FileManager.default.createDirectory(at: frames,
                                                    withIntermediateDirectories: true)
        } catch {
            print("[FineTune] mkdir failed: \(error)")
            return nil
        }
        self.runDirectory = runDir
        self.framesDir = frames
        self.labelsURL = runDir.appendingPathComponent("labels.csv")

        // Seed labels.csv with the header.
        let header = [
            "trial_idx", "frame_idx", "png_filename", "cell_idx",
            "cell_cx_pt", "cell_cy_pt",
            "Rn_00", "Rn_01", "Rn_02",
            "Rn_10", "Rn_11", "Rn_12",
            "Rn_20", "Rn_21", "Rn_22",
            "eye_cam_x_mm", "eye_cam_y_mm", "eye_cam_z_mm",
            "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
            "head_tx_mm", "head_ty_mm", "head_tz_mm",
            "gaze_pred_cam_x", "gaze_pred_cam_y", "gaze_pred_cam_z",
            "pupil_left_px", "pupil_right_px"
        ].joined(separator: ",") + "\n"
        do {
            try header.write(to: labelsURL, atomically: true, encoding: .utf8)
            self.labelsHandle = try FileHandle(forWritingTo: labelsURL)
            try self.labelsHandle?.seekToEnd()
        } catch {
            print("[FineTune] labels init failed: \(error)")
            return nil
        }

        Self.writeMeta(to: runDir,
                       screenSize: screenSize,
                       intrinsics: intrinsics)
        Self.writeCalibration(to: runDir, calibration: calibration)
    }

    deinit {
        try? labelsHandle?.close()
    }

    /// Persist one frame: face PNG + one row in `labels.csv`.
    func write(trialIdx: Int,
               cellIdx: Int,
               cellCenter: CGPoint,
               faceImage: UIImage,
               normalizationRotation Rn: simd_double3x3,
               eyePositionCam: simd_double3,
               headPose: HeadPose,
               gazePredCam: simd_double3,
               pupilDiameters: PupilMeasure.Diameters) {
        let frameIdx = frameCount
        frameCount += 1
        let pngName = String(format: "%02d_%05d.png", trialIdx, frameIdx)
        let pngURL = framesDir.appendingPathComponent(pngName)
        if let data = faceImage.pngData() {
            try? data.write(to: pngURL, options: .atomic)
        }

        // simd column-major: Rn.columns.c[row] gives Rn[row, col] in math
        // notation. Emit row-major so the laptop can read [row][col]
        // without re-thinking the convention.
        let r00 = Rn.columns.0.x, r01 = Rn.columns.1.x, r02 = Rn.columns.2.x
        let r10 = Rn.columns.0.y, r11 = Rn.columns.1.y, r12 = Rn.columns.2.y
        let r20 = Rn.columns.0.z, r21 = Rn.columns.1.z, r22 = Rn.columns.2.z

        let e = headPose.euler
        let row: [String] = [
            "\(trialIdx)", "\(frameIdx)", pngName, "\(cellIdx)",
            fmt(Double(cellCenter.x), 2), fmt(Double(cellCenter.y), 2),
            fmt(r00, 6), fmt(r01, 6), fmt(r02, 6),
            fmt(r10, 6), fmt(r11, 6), fmt(r12, 6),
            fmt(r20, 6), fmt(r21, 6), fmt(r22, 6),
            fmt(eyePositionCam.x, 3), fmt(eyePositionCam.y, 3), fmt(eyePositionCam.z, 3),
            fmt(e.yaw, 3), fmt(e.pitch, 3), fmt(e.roll, 3),
            fmt(headPose.translation.x, 3), fmt(headPose.translation.y, 3), fmt(headPose.translation.z, 3),
            fmt(gazePredCam.x, 6), fmt(gazePredCam.y, 6), fmt(gazePredCam.z, 6),
            fmt(pupilDiameters.leftPx, 3), fmt(pupilDiameters.rightPx, 3)
        ]
        let line = row.joined(separator: ",") + "\n"
        if let data = line.data(using: .utf8) {
            try? labelsHandle?.write(contentsOf: data)
        }
    }

    /// Close `labels.csv` and zip the run directory. Returns the zip URL,
    /// or nil if zipping failed.
    func finalize() -> URL? {
        try? labelsHandle?.close()
        labelsHandle = nil

        let parent = runDirectory.deletingLastPathComponent()
        let zipURL = parent.appendingPathComponent("\(runName).zip")
        try? FileManager.default.removeItem(at: zipURL)

        let coordinator = NSFileCoordinator()
        var nsErr: NSError?
        var success = false
        coordinator.coordinate(readingItemAt: runDirectory,
                               options: [.forUploading],
                               error: &nsErr) { tmpZip in
            do {
                try FileManager.default.copyItem(at: tmpZip, to: zipURL)
                success = true
            } catch {
                print("[FineTune] zip copy failed: \(error)")
            }
        }
        if let e = nsErr {
            print("[FineTune] coordinator error: \(e)")
        }
        return success ? zipURL : nil
    }

    // MARK: - Static helpers

    private static func writeMeta(to runDir: URL,
                                  screenSize: CGSize,
                                  intrinsics: CameraIntrinsics) {
        let meta: [String: Any] = [
            "schema_version": 1,
            "screen_w_pt": screenSize.width,
            "screen_h_pt": screenSize.height,
            "intrinsics": [
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
                "image_w": intrinsics.imageWidth,
                "image_h": intrinsics.imageHeight
            ],
            "face_normalizer": [
                "output_w": FaceNormalizer.outputWidth,
                "output_h": FaceNormalizer.outputHeight,
                "normalized_distance_mm": FaceNormalizer.normalizedDistance,
                "normalized_focal_px": FaceNormalizer.normalizedFocal,
                "horizontal_flip": true
            ],
            "eth_xgaze_convention": [
                "gaze_norm_formula": "g_eth = (cos(p)*sin(y), sin(p), cos(p)*cos(y))",
                "wrapper_imagenet_normalize_inputs": true
            ]
        ]
        if let data = try? JSONSerialization.data(withJSONObject: meta,
                                                  options: [.prettyPrinted]) {
            try? data.write(to: runDir.appendingPathComponent("meta.json"),
                            options: .atomic)
        }
    }

    private static func writeCalibration(to runDir: URL,
                                         calibration cal: CalibrationModel) {
        let t = cal.translation
        let targets: [[Double]] = cal.targets.map { [$0.x, $0.y] }
        let offsets: [[Double]] = cal.offsets.map { [$0.x, $0.y] }
        let obj: [String: Any] = [
            "translation": [t.x, t.y, t.z],
            "targets": targets,
            "offsets": offsets
        ]
        if let data = try? JSONSerialization.data(withJSONObject: obj,
                                                  options: [.prettyPrinted]) {
            try? data.write(to: runDir.appendingPathComponent("calibration.json"),
                            options: .atomic)
        }
    }

    private func fmt(_ x: Double, _ places: Int) -> String {
        guard x.isFinite else { return "" }
        return String(format: "%.\(places)f", x)
    }
}
