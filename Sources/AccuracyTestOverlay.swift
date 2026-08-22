import SwiftUI
import simd

/// Full-screen overlay shown during an accuracy-test run. Draws one red
/// target at a time; the user has already calibrated, so we don't want any
/// other on-screen distractions (including the live gaze dot, which
/// `ContentView` hides while the controller is active).
struct AccuracyTestOverlay: View {
    @ObservedObject var controller: AccuracyTestController
    let screenSize: CGSize
    let onCancel: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            let target = controller.currentTarget
            let center = CGPoint(
                x: screenSize.width  * 0.5 + CGFloat(target.x),
                y: screenSize.height * 0.5 + CGFloat(target.y)
            )
            let isCapturing = controller.phase == .capturing
            ZStack {
                Circle()
                    .stroke(Color.white.opacity(0.35), lineWidth: 2)
                    .frame(width: 44, height: 44)
                Circle()
                    .fill(Color.red)
                    .frame(width: isCapturing ? 22 : 18,
                           height: isCapturing ? 22 : 18)
                if isCapturing {
                    Circle()
                        .stroke(Color.white, lineWidth: 2)
                        .frame(width: 30, height: 30)
                }
            }
            .position(center)

            VStack(spacing: 10) {
                Text(headlineText)
                    .font(.headline)
                    .foregroundColor(.white)
                Text("Point \(controller.dotIndex + 1) of \(controller.targets.count)")
                    .font(.caption.monospaced())
                    .foregroundColor(.white.opacity(0.7))
                ProgressView(value: controller.dotProgress)
                    .progressViewStyle(.linear)
                    .frame(width: 220)
                    .tint(isCapturing ? .red : .white)
            }
            .frame(maxHeight: .infinity, alignment: .top)
            .padding(.top, 80)

            Button(action: onCancel) {
                Text("Cancel")
                    .font(.body.weight(.medium))
                    .padding(.horizontal, 28)
                    .padding(.vertical, 10)
                    .background(Color.white.opacity(0.18))
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            .frame(maxHeight: .infinity, alignment: .bottom)
            .padding(.bottom, 60)
        }
    }

    private var headlineText: String {
        switch controller.phase {
        case .idle:           return ""
        case .dwelling:       return "Look at the red dot…"
        case .capturing:      return "Hold steady — measuring"
        case .complete:       return "Done!"
        case .failed(let m):  return "Failed: \(m)"
        }
    }
}
