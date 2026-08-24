import Foundation
import CoreML
import simd

/// Stage 4 of the gaze pipeline: predicts gaze direction from a normalized
/// face image.
///
/// ## Expected CoreML model — ETH-XGaze ResNet18
///
/// Bundled as `GazeNet.mlpackage` (Xcode compiles to `GazeNet.mlmodelc` at
/// build time). Produced by `Tools/convert_eth_xgaze_to_coreml.py` from
/// hysts' `pl_gaze_estimation` checkpoint
/// (`eth-xgaze_resnet18.pth`). The conversion script wraps the network so
/// ImageNet normalization is baked into the model — Swift just sends RGB
/// floats in `[0, 1]`.
///
/// **Input**  — `MLMultiArray`, shape `[1, 3, 224, 224]` (NCHW), Float32, in
/// `[0, 1]`. Channel order is **RGB**. We accept the matching tensor from
/// `FaceNormalizer.Output` (HWC) and transpose during MLMultiArray
/// construction.
///
/// **Output** — `MLMultiArray`, shape `[1, 2]` or `[2]`, Float32, holding
/// `(pitch, yaw)` in **radians** in that order. ETH-XGaze gaze convention
/// (per `pl_gaze_estimation/common/face_parts.py`):
/// ```
///   gaze_norm_eth = -(cos(p)·sin(y), sin(p), cos(p)·cos(y))
/// ```
/// — a unit vector in the *normalized face* coordinate frame, where +X is
/// the head's anatomical-RIGHT direction.
///
/// ## Frame conventions
///
/// Our pipeline normalizes a face image whose horizontal axis is
/// **anatomical-LEFT** (because the front camera buffer is mirrored AND
/// our 3D face model has X flipped — see `CanonicalFaceModel`).
/// `FaceNormalizer` therefore horizontally flips its output before we run
/// the CNN, so the network sees a canonical un-mirrored face. The X-flip is
/// undone here: after computing `gaze_eth_norm`, we negate `x` to express
/// the same direction in our (mirrored) normalized frame, then rotate back
/// to camera coords via `R_n^T`.
final class GazeEstimator {

    struct Estimate {
        /// Gaze pitch in radians (positive = looking up).
        let pitch: Double
        /// Gaze yaw in radians (positive = looking right in ETH-XGaze frame).
        let yaw: Double
        /// Unit gaze vector in the normalized face frame (ours, post X-undo).
        let gazeNorm: simd_double3
        /// Unit gaze vector in the original camera frame.
        let gazeCam: simd_double3
    }

    /// Tensor / MLMultiArray spatial dimensions. ETH-XGaze ResNet18 takes
    /// 224×224. If the model is ever retrained at a different resolution,
    /// update both these constants and `FaceNormalizer.outputWidth/height`.
    static let inputHeight = 224
    static let inputWidth  = 224

    /// Resource name (without extension) we look for in the main bundle.
    private static let modelResourceName = "GazeNet"

    private var model: MLModel?
    private var inputName: String?
    /// True iff a model is currently loaded.
    var isModelLoaded: Bool { model != nil }
    /// Human-readable description of the currently-loaded weights (bundled
    /// path or the hot-swap URL the user fetched). Useful for the HUD.
    private(set) var modelSource: String = "—"

    init() {
        // Xcode compiles `.mlpackage` to `.mlmodelc` at build time, so look
        // up by `mlmodelc` extension. Keep the lookup forgiving: if a future
        // build process leaves a `.mlpackage` instead, that still works.
        if let url = Bundle.main.url(
                forResource: Self.modelResourceName, withExtension: "mlmodelc")
            ?? Bundle.main.url(
                forResource: Self.modelResourceName, withExtension: "mlpackage"),
           let m = try? MLModel(contentsOf: url) {
            self.model = m
            self.inputName = m.modelDescription.inputDescriptionsByName.keys.first
            self.modelSource = "bundled"
            print("[Gaze] Stage 4: loaded \(Self.modelResourceName) " +
                  "(input=\(self.inputName ?? "?")).")
        } else {
            self.model = nil
            self.inputName = nil
            print("[Gaze] Stage 4: no \(Self.modelResourceName) bundled — " +
                  "gaze prediction disabled.")
        }
    }

    /// Hot-swap in a freshly-compiled model. The caller is responsible for
    /// running `MLModel.compileModel(at:)` on a background queue first —
    /// keeping that off the main actor avoids both the watchdog timeout and
    /// the memory spike that triggers iOS jetsam when the camera + MediaPipe
    /// are still running.
    ///
    /// Errors leave the previously-loaded model untouched.
    @discardableResult
    func replace(compiledModelAt compiledURL: URL, sourceLabel: String) throws -> String {
        let newModel = try MLModel(contentsOf: compiledURL)
        self.model = newModel
        self.inputName = newModel.modelDescription.inputDescriptionsByName.keys.first
        self.modelSource = sourceLabel
        let label = self.inputName ?? "?"
        print("[Gaze] Stage 4: hot-swapped to \(sourceLabel) (input=\(label)).")
        return label
    }

    /// Run the CNN on a 224·224·3 RGB float tensor and convert to a 3D gaze
    /// vector in camera space. `tensorRGB` and `normalizationRotation` come
    /// straight from `FaceNormalizer.Output`.
    func estimate(tensorRGB: [Float],
                  normalizationRotation Rn: simd_double3x3) -> Estimate? {
        guard let model = model, let inputName = inputName else { return nil }
        let H = Self.inputHeight, W = Self.inputWidth
        guard tensorRGB.count == H * W * 3 else { return nil }

        guard let input = Self.makeNCHWInput(from: tensorRGB, H: H, W: W)
        else { return nil }
        let provider: MLFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(
                dictionary: [inputName: MLFeatureValue(multiArray: input)]
            )
        } catch {
            print("[Gaze] Stage 4: input feature error: \(error)")
            return nil
        }

        let result: MLFeatureProvider
        do {
            result = try model.prediction(from: provider)
        } catch {
            print("[Gaze] Stage 4: prediction error: \(error)")
            return nil
        }

        guard let outName = model.modelDescription
                .outputDescriptionsByName.keys.first,
              let outArr = result.featureValue(for: outName)?.multiArrayValue,
              outArr.count >= 2 else {
            return nil
        }

        let pitch = outArr[0].doubleValue
        let yaw   = outArr[1].doubleValue
        return Self.estimate(pitch: pitch, yaw: yaw, normalizationRotation: Rn)
    }

    /// Build an `Estimate` from `(pitch, yaw)` and the normalization rotation
    /// the crop was taken with, without touching the CNN.
    ///
    /// Extracted from `estimate(tensorRGB:normalizationRotation:)` so the
    /// One Euro smoother can filter the two angles the network actually
    /// predicts and then re-derive the camera-frame ray from them. Smoothing
    /// `gazeCam` directly instead would smooth a unit vector through
    /// non-linear terms and would not stay unit-length.
    static func estimate(pitch: Double, yaw: Double,
                         normalizationRotation Rn: simd_double3x3) -> Estimate {
        // ETH-XGaze convention: gaze_norm_eth = -(cos(p)·sin(y), sin(p), cos(p)·cos(y))
        let cp = cos(pitch), sp = sin(pitch)
        let cy = cos(yaw),   sy = sin(yaw)
        let gazeEthNorm = simd_double3(-cp * sy, -sp, -cp * cy)
        // Undo the horizontal flip applied in FaceNormalizer: negate X to
        // express the same direction in our (mirrored) normalized frame.
        let gazeOurNorm = simd_double3(-gazeEthNorm.x,
                                        gazeEthNorm.y,
                                        gazeEthNorm.z)
        // R_n^T maps a vector from our normalized space back to camera space.
        let gazeCam = simd_normalize(Rn.transpose * gazeOurNorm)

        return Estimate(pitch: pitch, yaw: yaw,
                        gazeNorm: gazeOurNorm, gazeCam: gazeCam)
    }

    /// HWC `[Float]` (H·W·3, RGB, [0,1]) → NCHW `MLMultiArray` `[1, 3, H, W]`.
    private static func makeNCHWInput(from hwc: [Float],
                                       H: Int, W: Int) -> MLMultiArray? {
        let C = 3
        guard let arr = try? MLMultiArray(
            shape: [1, NSNumber(value: C),
                       NSNumber(value: H), NSNumber(value: W)],
            dataType: .float32
        ) else { return nil }

        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        let strideC = arr.strides[1].intValue
        let strideH = arr.strides[2].intValue
        let strideW = arr.strides[3].intValue

        hwc.withUnsafeBufferPointer { src in
            let s = src.baseAddress!
            for y in 0..<H {
                for x in 0..<W {
                    let srcBase = (y * W + x) * C
                    for c in 0..<C {
                        dst[c * strideC + y * strideH + x * strideW] =
                            s[srcBase + c]
                    }
                }
            }
        }
        return arr
    }
}
