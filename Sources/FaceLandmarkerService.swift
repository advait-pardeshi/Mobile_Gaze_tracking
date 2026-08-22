import Foundation
import MediaPipeTasksVision
import AVFoundation
import UIKit

protocol FaceLandmarkerServiceDelegate: AnyObject {
    func faceLandmarkerService(_ service: FaceLandmarkerService,
                               didDetect result: FaceLandmarkerResult,
                               timestampMs: Int)
    func faceLandmarkerService(_ service: FaceLandmarkerService,
                               didFailWith error: Error)
}

final class FaceLandmarkerService: NSObject {
    weak var delegate: FaceLandmarkerServiceDelegate?

    private var faceLandmarker: FaceLandmarker?

    init?(modelName: String = "face_landmarker") {
        super.init()
        guard let modelPath = Bundle.main.path(forResource: modelName, ofType: "task") else {
            print("[FaceLandmarker] Missing \(modelName).task in bundle.")
            return nil
        }
        let options = FaceLandmarkerOptions()
        options.baseOptions.modelAssetPath = modelPath
        options.runningMode = .liveStream
        options.numFaces = 1
        options.minFaceDetectionConfidence = 0.5
        options.minFacePresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5
        options.outputFaceBlendshapes = false
        options.outputFacialTransformationMatrixes = false
        options.faceLandmarkerLiveStreamDelegate = self

        do {
            faceLandmarker = try FaceLandmarker(options: options)
        } catch {
            print("[FaceLandmarker] Init failed: \(error)")
            return nil
        }
    }

    func detectAsync(sampleBuffer: CMSampleBuffer,
                     orientation: UIImage.Orientation,
                     timestampMs: Int) {
        guard let faceLandmarker = faceLandmarker else { return }
        do {
            let image = try MPImage(sampleBuffer: sampleBuffer, orientation: orientation)
            try faceLandmarker.detectAsync(image: image, timestampInMilliseconds: timestampMs)
        } catch {
            delegate?.faceLandmarkerService(self, didFailWith: error)
        }
    }
}

extension FaceLandmarkerService: FaceLandmarkerLiveStreamDelegate {
    func faceLandmarker(_ faceLandmarker: FaceLandmarker,
                        didFinishDetection result: FaceLandmarkerResult?,
                        timestampInMilliseconds: Int,
                        error: Error?) {
        if let error = error {
            delegate?.faceLandmarkerService(self, didFailWith: error)
            return
        }
        guard let result = result else { return }
        delegate?.faceLandmarkerService(self,
                                        didDetect: result,
                                        timestampMs: timestampInMilliseconds)
    }
}
