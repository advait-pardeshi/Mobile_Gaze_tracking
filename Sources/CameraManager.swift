import AVFoundation
import UIKit

protocol CameraManagerDelegate: AnyObject {
    func cameraManager(_ manager: CameraManager,
                       didOutput sampleBuffer: CMSampleBuffer,
                       orientation: UIImage.Orientation)
}

final class CameraManager: NSObject {
    let session = AVCaptureSession()
    weak var delegate: CameraManagerDelegate?

    private(set) var imageWidth: Int = 0
    private(set) var imageHeight: Int = 0
    /// Set once `configure()` finishes. Read by the head-pose stage on the
    /// camera queue (single-writer, eventual-consistency reader).
    private(set) var intrinsics: CameraIntrinsics?

    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "gaze.camera.session")
    private let videoQueue  = DispatchQueue(label: "gaze.camera.video")
    private var configured = false

    func requestAuthorization(_ completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async { completion(granted) }
            }
        default:
            completion(false)
        }
    }

    func configure() {
        sessionQueue.async {
            guard !self.configured else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .hd1280x720

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                       for: .video,
                                                       position: .front),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                return
            }
            self.session.addInput(input)

            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            self.videoOutput.setSampleBufferDelegate(self, queue: self.videoQueue)
            if self.session.canAddOutput(self.videoOutput) {
                self.session.addOutput(self.videoOutput)
            }

            if let connection = self.videoOutput.connection(with: .video) {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
                if connection.isVideoMirroringSupported {
                    connection.isVideoMirrored = true
                }
            }

            // Capture stream dimensions (after orientation: width=720, height=1280)
            let dims = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
            // After portrait rotation, width/height swap.
            self.imageWidth  = Int(dims.height)
            self.imageHeight = Int(dims.width)
            self.intrinsics = CameraIntrinsics.make(
                device: device,
                portraitWidth: self.imageWidth,
                portraitHeight: self.imageHeight
            )

            self.session.commitConfiguration()
            self.configured = true
        }
    }

    func start() {
        sessionQueue.async {
            if !self.session.isRunning { self.session.startRunning() }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        // Buffer is already rotated to portrait + horizontally mirrored (front camera),
        // so MediaPipe should treat it as upright.
        delegate?.cameraManager(self, didOutput: sampleBuffer, orientation: .up)
    }
}
