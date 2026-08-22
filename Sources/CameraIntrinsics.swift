import Foundation
import AVFoundation
import CoreMedia

/// Pinhole camera intrinsics in pixels for the **delivered (portrait, mirrored) frame**.
///
/// Derived once from the active format's horizontal field-of-view. We don't
/// pull `kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix` per-frame because
/// that matrix is for the native sensor orientation (landscape) and would
/// need to be rotated/mirrored to match our delivered buffer — error-prone
/// for a small accuracy gain.
struct CameraIntrinsics {
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double
    let imageWidth: Double
    let imageHeight: Double

    /// Build from the device's active format and the **portrait** image dimensions.
    /// `device.activeFormat.videoFieldOfView` is the horizontal FOV of the
    /// landscape sensor; in portrait, that becomes the vertical FOV.
    static func make(device: AVCaptureDevice,
                     portraitWidth: Int,
                     portraitHeight: Int) -> CameraIntrinsics {
        let sensorHFOVDeg = Double(device.activeFormat.videoFieldOfView)
        let sensorDims = CMVideoFormatDescriptionGetDimensions(
            device.activeFormat.formatDescription
        )
        let sensorAspect = Double(sensorDims.width) / Double(sensorDims.height)

        let sensorHFOVRad = sensorHFOVDeg * .pi / 180.0
        let sensorVFOVRad = 2.0 * atan(tan(sensorHFOVRad / 2.0) / sensorAspect)

        // In portrait, the image's horizontal axis is the sensor's vertical.
        let portraitHFOVRad = sensorVFOVRad

        let w = Double(portraitWidth)
        let h = Double(portraitHeight)

        let fx = (w / 2.0) / tan(portraitHFOVRad / 2.0)
        // Square-pixel assumption (true for iPhone front cameras): fy = fx.
        let fy = fx

        return CameraIntrinsics(
            fx: fx, fy: fy,
            cx: w / 2.0, cy: h / 2.0,
            imageWidth: w, imageHeight: h
        )
    }
}
