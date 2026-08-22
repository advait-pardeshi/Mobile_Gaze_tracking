import Foundation
import simd
import CoreVideo
import CoreGraphics
import UIKit

/// Phase 4 face normalization for the ETH-XGaze ResNet18 (Stage 4 CNN).
///
/// Same virtual-camera trick as `EyeNormalizer`, but anchored at the face
/// center and producing a 224×224 RGB image instead of a 100×50 eye crop:
///
///   - Look-at point = `CanonicalFaceModel.faceCenter3D` (mid-face, between
///     the eyes and slightly down toward the nose alar — matches the plgaze
///     definition for ETH-XGaze: mean of mp landmarks 33, 133, 362, 263,
///     240, 460).
///   - Virtual focal length = 960 px @ d = 600 mm (the values baked into
///     `data/normalized_camera_params/eth-xgaze.yaml`); scaled per-frame
///     by `f_n = focalNorm · d / dNorm` so the face fills a consistent area
///     of the 224×224 image regardless of how far the user is.
///   - Inverse-warp homography `W⁻¹ = K · R_n^T · K_n⁻¹`, identical math to
///     `EyeNormalizer`.
///
/// **Mirror handling.** Our front-camera buffer is `isVideoMirrored = true`
/// and our 3D model has `X` flipped (see `CanonicalFaceModel`). The
/// composed effect is a normalized image with `+X = anatomical-LEFT`, while
/// ETH-XGaze was trained with `+X = anatomical-RIGHT`. We therefore
/// **horizontally flip** the normalized image before producing the CNN
/// tensor. The CNN then sees a canonical un-mirrored face. The X-flip is
/// undone in `GazeEstimator` when mapping the predicted gaze back to camera
/// coordinates (one extra `X_flip` factor in the rotation chain).
final class FaceNormalizer {

    static let outputWidth  = 224
    static let outputHeight = 224

    /// Match plgaze's `data/normalized_camera_params/eth-xgaze.yaml`.
    static let normalizedDistance: Double = 600.0   // mm
    static let normalizedFocal:    Double = 960.0   // px @ normalizedDistance

    /// ImageNet normalization is baked into the CoreML model wrapper, so
    /// the tensor we emit is plain RGB float in `[0, 1]` (NOT pre-whitened).

    struct Output {
        /// 224×224 BGRA visualization (post-flip) — useful for debug overlays.
        let image: UIImage
        /// 224·224·3 floats in row-major (y, x, channel = RGB) order, [0,1].
        /// Ready for `GazeEstimator` to pack as `MLMultiArray [1, 3, 224, 224]`.
        let tensorRGB: [Float]
        /// Virtual-camera rotation `R_n` for the face. `R_n^T · v_norm = v_cam`.
        let rotation: simd_double3x3
    }

    func normalize(pixelBuffer: CVPixelBuffer,
                   headPose: HeadPose,
                   intrinsics: CameraIntrinsics) -> Output? {

        // 1. Face center in camera coords. Reject degenerate (face behind the
        // camera or numerically too close).
        let cFace = headPose.rotation * CanonicalFaceModel.faceCenter3D + headPose.translation
        guard cFace.z > 50 else { return nil }

        // 2. Virtual-camera rotation built from the head's own X axis so the
        // result is roll-stabilized (face stays level in the normalized frame).
        let Rn = Self.normalizationRotation(faceCam: cFace,
                                            headRotation: headPose.rotation)

        // 3. Lock buffer once and warp into a 224×224 BGRA scratch.
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        let srcPtr = base.assumingMemoryBound(to: UInt8.self)

        var bgra = [UInt8](repeating: 0,
                            count: Self.outputWidth * Self.outputHeight * 4)
        Self.warp(src: srcPtr, srcW: srcW, srcH: srcH, bytesPerRow: bytesPerRow,
                  faceCam: cFace, Rn: Rn, intrinsics: intrinsics, dst: &bgra)

        // 4. Horizontal flip — see file header for the mirror argument.
        let flipped = Self.horizontalFlip(bgra: bgra,
                                          width: Self.outputWidth,
                                          height: Self.outputHeight)

        guard let img = Self.makeImage(bgra: flipped,
                                       width: Self.outputWidth,
                                       height: Self.outputHeight)
        else { return nil }

        let tensor = Self.makeRGBTensor(bgra: flipped,
                                         width: Self.outputWidth,
                                         height: Self.outputHeight)

        return Output(image: img, tensorRGB: tensor, rotation: Rn)
    }

    // MARK: - Math helpers

    /// Build R_n the same way `EyeNormalizer` does, but with the face center
    /// as the look-at target.
    static func normalizationRotation(faceCam: simd_double3,
                                       headRotation: simd_double3x3) -> simd_double3x3 {
        let d = simd_length(faceCam)
        let z = faceCam / d
        let headX = headRotation[0]
        var yRaw = simd_cross(z, headX)
        let yLen = simd_length(yRaw)
        if yLen < 1e-6 {
            yRaw = simd_cross(z, simd_double3(0, 1, 0))
        }
        let y = simd_normalize(yRaw)
        let x = simd_cross(y, z)
        return simd_double3x3(columns: (
            simd_double3(x.x, y.x, z.x),
            simd_double3(x.y, y.y, z.y),
            simd_double3(x.z, y.z, z.z)
        ))
    }

    /// Per-pixel inverse warp + bilinear sample. Identical core to
    /// `EyeNormalizer.warp`; broken out so the two stages can evolve
    /// independently (face normalization may want different focal scaling
    /// or distortion handling later).
    static func warp(src: UnsafePointer<UInt8>,
                     srcW: Int, srcH: Int, bytesPerRow: Int,
                     faceCam: simd_double3,
                     Rn: simd_double3x3,
                     intrinsics: CameraIntrinsics,
                     dst: inout [UInt8]) {

        let d = simd_length(faceCam)
        let fN  = normalizedFocal * d / normalizedDistance
        let cxN = Double(outputWidth)  / 2.0
        let cyN = Double(outputHeight) / 2.0

        let K = simd_double3x3(columns: (
            simd_double3(intrinsics.fx, 0, 0),
            simd_double3(0, intrinsics.fy, 0),
            simd_double3(intrinsics.cx, intrinsics.cy, 1)
        ))
        let KnInv = simd_double3x3(columns: (
            simd_double3(1.0 / fN, 0, 0),
            simd_double3(0, 1.0 / fN, 0),
            simd_double3(-cxN / fN, -cyN / fN, 1)
        ))
        let Winv = K * Rn.transpose * KnInv

        let maxX = Double(srcW) - 1.000001
        let maxY = Double(srcH) - 1.000001

        dst.withUnsafeMutableBufferPointer { dstBuf in
            let dstPtr = dstBuf.baseAddress!
            for v in 0..<outputHeight {
                for u in 0..<outputWidth {
                    let p  = simd_double3(Double(u) + 0.5, Double(v) + 0.5, 1.0)
                    let q  = Winv * p
                    let iz = 1.0 / q.z
                    let sx = q.x * iz
                    let sy = q.y * iz

                    let outIdx = (v * outputWidth + u) * 4
                    if sx < 0 || sy < 0 || sx > maxX || sy > maxY {
                        dstPtr[outIdx]     = 0
                        dstPtr[outIdx + 1] = 0
                        dstPtr[outIdx + 2] = 0
                        dstPtr[outIdx + 3] = 255
                        continue
                    }
                    let x0 = Int(sx)
                    let y0 = Int(sy)
                    let dx = sx - Double(x0)
                    let dy = sy - Double(y0)
                    let row0 = src.advanced(by: y0 * bytesPerRow + x0 * 4)
                    let row1 = src.advanced(by: (y0 + 1) * bytesPerRow + x0 * 4)
                    let w00 = (1 - dx) * (1 - dy)
                    let w10 = dx * (1 - dy)
                    let w01 = (1 - dx) * dy
                    let w11 = dx * dy

                    for c in 0..<3 {
                        let v00 = Double(row0[c])
                        let v10 = Double(row0[c + 4])
                        let v01 = Double(row1[c])
                        let v11 = Double(row1[c + 4])
                        let s = v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11
                        dstPtr[outIdx + c] = UInt8(min(255.0, max(0.0, s)))
                    }
                    dstPtr[outIdx + 3] = 255
                }
            }
        }
    }

    /// Horizontal mirror in BGRA. Reverses each row in-place into a fresh buffer.
    static func horizontalFlip(bgra: [UInt8], width w: Int, height h: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: bgra.count)
        bgra.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<h {
                    let rowBase = y * w * 4
                    for x in 0..<w {
                        let s = rowBase + x * 4
                        let d = rowBase + (w - 1 - x) * 4
                        dst[d]     = src[s]
                        dst[d + 1] = src[s + 1]
                        dst[d + 2] = src[s + 2]
                        dst[d + 3] = src[s + 3]
                    }
                }
            }
        }
        return out
    }

    /// BGRA UInt8 → RGB float32 in `[0, 1]`, row-major (y, x, channel).
    static func makeRGBTensor(bgra: [UInt8], width w: Int, height h: Int) -> [Float] {
        var out = [Float](repeating: 0, count: w * h * 3)
        let inv: Float = 1.0 / 255.0
        for v in 0..<h {
            for u in 0..<w {
                let i = (v * w + u) * 4
                let o = (v * w + u) * 3
                out[o]     = Float(bgra[i + 2]) * inv  // R
                out[o + 1] = Float(bgra[i + 1]) * inv  // G
                out[o + 2] = Float(bgra[i])     * inv  // B
            }
        }
        return out
    }

    static func makeImage(bgra: [UInt8], width: Int, height: Int) -> UIImage? {
        let bytesPerRow = width * 4
        guard let provider = CGDataProvider(data: Data(bgra) as CFData) else { return nil }
        let bitmapInfo = CGBitmapInfo(rawValue:
            CGImageAlphaInfo.premultipliedFirst.rawValue |
            CGBitmapInfo.byteOrder32Little.rawValue
        )
        guard let cg = CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else { return nil }
        return UIImage(cgImage: cg)
    }
}
