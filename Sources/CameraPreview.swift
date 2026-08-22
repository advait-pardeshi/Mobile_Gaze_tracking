import SwiftUI
import AVFoundation
import MediaPipeTasksVision
import simd

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    let landmarks: [NormalizedLandmark]
    let imageSize: CGSize
    let headPose: HeadPose?
    let intrinsics: CameraIntrinsics?
    let gazeOriginCam: simd_double3?
    let gazeDirCam: simd_double3?

    func makeUIView(context: Context) -> PreviewContainerView {
        let view = PreviewContainerView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        if let conn = view.videoPreviewLayer.connection,
           conn.isVideoOrientationSupported {
            conn.videoOrientation = .portrait
        }
        return view
    }

    func updateUIView(_ uiView: PreviewContainerView, context: Context) {
        uiView.update(
            landmarks: landmarks,
            imageSize: imageSize,
            headPose: headPose,
            intrinsics: intrinsics,
            gazeOriginCam: gazeOriginCam,
            gazeDirCam: gazeDirCam
        )
    }
}

final class PreviewContainerView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        // swiftlint:disable:next force_cast
        layer as! AVCaptureVideoPreviewLayer
    }

    private let landmarkLayer = CAShapeLayer()
    private let axisXLayer = CAShapeLayer()
    private let axisYLayer = CAShapeLayer()
    private let axisZLayer = CAShapeLayer()
    private let gazeLayer = CAShapeLayer()

    private var currentLandmarks: [NormalizedLandmark] = []
    private var currentImageSize: CGSize = .zero
    private var currentHeadPose: HeadPose?
    private var currentIntrinsics: CameraIntrinsics?
    private var currentGazeOrigin: simd_double3?
    private var currentGazeDir:    simd_double3?

    override init(frame: CGRect) {
        super.init(frame: frame)
        configureOverlay()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        configureOverlay()
    }

    private func configureOverlay() {
        landmarkLayer.fillColor = UIColor.green.cgColor
        landmarkLayer.strokeColor = UIColor.clear.cgColor
        landmarkLayer.zPosition = 10
        layer.addSublayer(landmarkLayer)

        for (axisLayer, color) in [
            (axisXLayer, UIColor.systemRed),
            (axisYLayer, UIColor.systemGreen),
            (axisZLayer, UIColor.systemBlue),
        ] {
            axisLayer.fillColor = UIColor.clear.cgColor
            axisLayer.strokeColor = color.cgColor
            axisLayer.lineWidth = 4
            axisLayer.lineCap = .round
            axisLayer.zPosition = 11
            layer.addSublayer(axisLayer)
        }

        gazeLayer.fillColor = UIColor.clear.cgColor
        gazeLayer.strokeColor = UIColor.systemYellow.cgColor
        gazeLayer.lineWidth = 4
        gazeLayer.lineCap = .round
        gazeLayer.zPosition = 12
        layer.addSublayer(gazeLayer)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        for l in [landmarkLayer, axisXLayer, axisYLayer, axisZLayer, gazeLayer] {
            l.frame = bounds
        }
        redraw()
    }

    func update(landmarks: [NormalizedLandmark],
                imageSize: CGSize,
                headPose: HeadPose?,
                intrinsics: CameraIntrinsics?,
                gazeOriginCam: simd_double3?,
                gazeDirCam: simd_double3?) {
        self.currentLandmarks = landmarks
        self.currentImageSize = imageSize
        self.currentHeadPose = headPose
        self.currentIntrinsics = intrinsics
        self.currentGazeOrigin = gazeOriginCam
        self.currentGazeDir = gazeDirCam
        redraw()
    }

    /// Image-pixel → view-pixel transform matching `.resizeAspectFill`.
    private func imageToView(_ p: CGPoint, imageSize: CGSize, viewSize: CGSize) -> CGPoint {
        let scale = max(viewSize.width / imageSize.width,
                        viewSize.height / imageSize.height)
        let scaledW = imageSize.width * scale
        let scaledH = imageSize.height * scale
        let dx = (viewSize.width  - scaledW) / 2
        let dy = (viewSize.height - scaledH) / 2
        return CGPoint(x: p.x / imageSize.width  * scaledW + dx,
                       y: p.y / imageSize.height * scaledH + dy)
    }

    private func redraw() {
        guard currentImageSize.width > 0,
              currentImageSize.height > 0,
              bounds.width > 0 else {
            landmarkLayer.path = nil
            axisXLayer.path = nil
            axisYLayer.path = nil
            axisZLayer.path = nil
            gazeLayer.path = nil
            return
        }

        let viewSize = bounds.size

        // ── Landmarks ─────────────────────────────────────────────
        if currentLandmarks.isEmpty {
            landmarkLayer.path = nil
        } else {
            let path = UIBezierPath()
            let r: CGFloat = 1.6
            for lm in currentLandmarks {
                let imgPt = CGPoint(
                    x: CGFloat(lm.x) * currentImageSize.width,
                    y: CGFloat(lm.y) * currentImageSize.height
                )
                let p = imageToView(imgPt, imageSize: currentImageSize, viewSize: viewSize)
                path.append(UIBezierPath(arcCenter: p, radius: r,
                                         startAngle: 0, endAngle: .pi * 2, clockwise: true))
            }
            landmarkLayer.path = path.cgPath
        }

        // ── Head-pose axis triad ─────────────────────────────────
        guard let pose = currentHeadPose, let K = currentIntrinsics else {
            axisXLayer.path = nil
            axisYLayer.path = nil
            axisZLayer.path = nil
            gazeLayer.path = nil
            return
        }

        // Project the 4 axis tips through (R, t, K) into image pixels,
        // then map to view pixels via the same aspect-fill transform.
        let projected: [CGPoint?] = CanonicalFaceModel.axisTips.map { p3 -> CGPoint? in
            let pc = pose.rotation * p3 + pose.translation
            guard pc.z > 1.0 else { return nil }
            let u = K.fx * pc.x / pc.z + K.cx
            let v = K.fy * pc.y / pc.z + K.cy
            return imageToView(CGPoint(x: u, y: v),
                               imageSize: currentImageSize, viewSize: viewSize)
        }
        guard let origin = projected[0],
              let xTip = projected[1],
              let yTip = projected[2],
              let zTip = projected[3] else {
            axisXLayer.path = nil
            axisYLayer.path = nil
            axisZLayer.path = nil
            return
        }

        for (axisLayer, tip) in [
            (axisXLayer, xTip),
            (axisYLayer, yTip),
            (axisZLayer, zTip),
        ] {
            let p = UIBezierPath()
            p.move(to: origin)
            p.addLine(to: tip)
            axisLayer.path = p.cgPath
        }

        // ── Gaze ray (Stage 4) ───────────────────────────────────
        // Project the eye-midpoint and a point ~200mm along the gaze direction
        // in camera space, then connect them in image space.
        guard let gOrigin = currentGazeOrigin,
              let gDir = currentGazeDir else {
            gazeLayer.path = nil
            return
        }
        let rayLengthMM = 200.0
        let p0 = gOrigin
        let p1 = gOrigin + gDir * rayLengthMM
        func project(_ p: simd_double3) -> CGPoint? {
            guard p.z > 1.0 else { return nil }
            let u = K.fx * p.x / p.z + K.cx
            let v = K.fy * p.y / p.z + K.cy
            return imageToView(CGPoint(x: u, y: v),
                               imageSize: currentImageSize, viewSize: viewSize)
        }
        guard let s = project(p0), let e = project(p1) else {
            gazeLayer.path = nil
            return
        }
        let path = UIBezierPath()
        path.move(to: s)
        path.addLine(to: e)
        // Tip dot so direction is unambiguous.
        path.append(UIBezierPath(arcCenter: e, radius: 5,
                                 startAngle: 0, endAngle: .pi * 2,
                                 clockwise: true))
        gazeLayer.path = path.cgPath
    }
}
