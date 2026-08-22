import Foundation
import simd
import CoreVideo
import CoreGraphics
import UIKit

/// Stage 3 of the gaze pipeline: head-pose-normalized eye crops.
///
/// For each eye we synthesize a *virtual camera* that
///   1. is rotated so its optical axis points directly at the eye center, and
///   2. has a focal length proportional to the current eye distance,
///      so the eye projects to a fixed pixel size regardless of how far
///      the user is from the phone.
///
/// The result is a 100×50 BGRA image per eye where the eye lands at the
/// principal point. Concatenated horizontally → 200×50, which is the
/// canonical input expected by the gaze CNN in Stage 4.
///
/// Math:
///   - Eye position in camera frame:  e = R_h · e_face + t_h
///   - Virtual camera basis:
///       z = e / ‖e‖                          (look at eye)
///       y = normalize(z × R_h·x_face)        (perpendicular to z and head's X)
///       x = y × z                            (right of virtual camera)
///   - R_n has rows (x, y, z) so that  v_norm = R_n · v_cam
///   - Virtual intrinsics:
///       f_n = focalNorm · d / dNorm          (eye stays same size in image)
///       K_n = [[f_n,0,W/2],[0,f_n,H/2],[0,0,1]]
///   - Output→source homography (used for inverse sampling):
///       W_inv = K · R_n^T · K_n^{-1}
final class EyeNormalizer {

    // Per-eye output. Total combined image is 2·outputWidth × outputHeight.
    static let outputWidth  = 100
    static let outputHeight = 50
    static let combinedWidth = outputWidth * 2

    // Virtual camera setup. focalNorm tunes how much of the face surrounds the
    // eye in the crop: bigger value = tighter crop on the eye itself.
    static let normalizedDistance: Double = 600.0  // mm
    static let normalizedFocal:    Double = 1000.0 // px @ normalizedDistance

    /// Output of one Stage 3 invocation.
    struct Output {
        /// 200×50 BGRA visualization, ready for an `Image(uiImage:)`.
        let combined: UIImage
        /// 50·200·3 floats in row-major (y, x, channel=RGB) order, in [0,1].
        /// Ready to feed Stage 4's CNN.
        let tensorRGB: [Float]
        /// Per-eye virtualcamera rotations. Stage 5 will need these to convert
        /// the gaze direction predicted in normalized space back into the
        /// original camera frame (gaze_cam = R_n^T · gaze_norm).
        let leftRotation:  simd_double3x3
        let rightRotation: simd_double3x3
    }

    func normalize(pixelBuffer: CVPixelBuffer,
                   headPose: HeadPose,
                   intrinsics: CameraIntrinsics) -> Output? {

        // 1. Eye centers in camera coords.
        let eL = headPose.rotation * CanonicalFaceModel.leftEyeCenter3D  + headPose.translation
        let eR = headPose.rotation * CanonicalFaceModel.rightEyeCenter3D + headPose.translation
        guard eL.z > 50, eR.z > 50 else { return nil }

        // 2. Per-eye virtual camera rotations.
        let RnL = Self.normalizationRotation(eyeCam: eL, headRotation: headPose.rotation)
        let RnR = Self.normalizationRotation(eyeCam: eR, headRotation: headPose.rotation)

        // 3. Lock buffer once, warp both eyes from the same source.
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        let srcPtr = base.assumingMemoryBound(to: UInt8.self)

        var leftBGRA  = [UInt8](repeating: 0, count: Self.outputWidth * Self.outputHeight * 4)
        var rightBGRA = [UInt8](repeating: 0, count: Self.outputWidth * Self.outputHeight * 4)
        Self.warp(src: srcPtr, srcW: srcW, srcH: srcH, bytesPerRow: bytesPerRow,
                  eyeCam: eL, Rn: RnL, intrinsics: intrinsics, dst: &leftBGRA)
        Self.warp(src: srcPtr, srcW: srcW, srcH: srcH, bytesPerRow: bytesPerRow,
                  eyeCam: eR, Rn: RnR, intrinsics: intrinsics, dst: &rightBGRA)

        // 4. Concatenate left | right → 200×50 BGRA.
        let combinedBGRA = Self.concatenate(left: leftBGRA, right: rightBGRA)
        guard let combinedImage = Self.makeImage(bgra: combinedBGRA,
                                                  width: Self.combinedWidth,
                                                  height: Self.outputHeight)
        else { return nil }

        // 5. CNN tensor: BGRA UInt8 → RGB float32 in [0,1].
        let tensor = Self.makeRGBTensor(bgra: combinedBGRA,
                                         width: Self.combinedWidth,
                                         height: Self.outputHeight)

        return Output(combined: combinedImage,
                      tensorRGB: tensor,
                      leftRotation: RnL,
                      rightRotation: RnR)
    }

    // MARK: - Math helpers

    /// Build R_n (camera→virtual-camera rotation) for a single eye.
    /// Rows of the returned matrix are the new basis vectors, expressed in
    /// the original camera frame.
    static func normalizationRotation(eyeCam: simd_double3,
                                      headRotation: simd_double3x3) -> simd_double3x3 {
        let d = simd_length(eyeCam)
        let z = eyeCam / d
        // Head's anatomical-X axis in camera coords. With the X-flipped canonical
        // model, this points toward image-right when the user faces the camera.
        let headX = headRotation[0]
        var yRaw = simd_cross(z, headX)
        let yLen = simd_length(yRaw)
        // Degenerate: head looking straight down the camera-X axis. Fall back
        // to using world up so we still produce a sane (if unrolled) frame.
        if yLen < 1e-6 {
            yRaw = simd_cross(z, simd_double3(0, 1, 0))
        }
        let y = simd_normalize(yRaw)
        let x = simd_cross(y, z)  // unit, since y ⊥ z and both are unit

        // simd_double3x3 is column-major: M[col][row]. We want rows (x, y, z),
        // so each column k holds (x.k, y.k, z.k).
        return simd_double3x3(columns: (
            simd_double3(x.x, y.x, z.x),
            simd_double3(x.y, y.y, z.y),
            simd_double3(x.z, y.z, z.z)
        ))
    }

    /// Per-pixel inverse-warp + bilinear sample for one eye. Writes BGRA bytes
    /// into `dst` (length must be outputWidth·outputHeight·4).
    static func warp(src: UnsafePointer<UInt8>,
                     srcW: Int, srcH: Int, bytesPerRow: Int,
                     eyeCam: simd_double3,
                     Rn: simd_double3x3,
                     intrinsics: CameraIntrinsics,
                     dst: inout [UInt8]) {

        let d = simd_length(eyeCam)
        let fN  = normalizedFocal * d / normalizedDistance
        let cxN = Double(outputWidth)  / 2.0
        let cyN = Double(outputHeight) / 2.0

        // K (original camera) and K_n^{-1} (virtual camera) in column-major form.
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
                    // Add 0.5 to sample at pixel centers.
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

                    // BGRA: channels 0..2 are colour, 3 is alpha.
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

    /// Place left|right side-by-side into a single 200×50 BGRA buffer.
    static func concatenate(left: [UInt8], right: [UInt8]) -> [UInt8] {
        let w = combinedWidth
        let h = outputHeight
        let halfW = outputWidth
        var out = [UInt8](repeating: 0, count: w * h * 4)
        let halfRow = halfW * 4
        for v in 0..<h {
            let dstRow = v * w * 4
            let srcRow = v * halfW * 4
            // Left half
            out.withUnsafeMutableBufferPointer { o in
                left.withUnsafeBufferPointer { l in
                    o.baseAddress!.advanced(by: dstRow)
                        .update(from: l.baseAddress!.advanced(by: srcRow), count: halfRow)
                }
                right.withUnsafeBufferPointer { r in
                    o.baseAddress!.advanced(by: dstRow + halfRow)
                        .update(from: r.baseAddress!.advanced(by: srcRow), count: halfRow)
                }
            }
        }
        return out
    }

    /// BGRA UInt8 → RGB float32 in [0,1], row-major (y, x, channel).
    static func makeRGBTensor(bgra: [UInt8], width: Int, height: Int) -> [Float] {
        var out = [Float](repeating: 0, count: width * height * 3)
        let inv: Float = 1.0 / 255.0
        for v in 0..<height {
            for u in 0..<width {
                let i = (v * width + u) * 4
                let o = (v * width + u) * 3
                out[o]     = Float(bgra[i + 2]) * inv  // R
                out[o + 1] = Float(bgra[i + 1]) * inv  // G
                out[o + 2] = Float(bgra[i])     * inv  // B
            }
        }
        return out
    }

    /// Wrap a BGRA byte array as a UIImage by way of CGImage. The buffer is
    /// treated as 32-bit pixels with the alpha byte first in memory order
    /// `B,G,R,A` (= little-endian ARGB with premultiplied-first alpha).
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
