import SwiftUI
import simd

/// Full-screen overlay shown during a Stage 5 calibration run. Draws one
/// dot at a time at the controller's current target, retints during the
/// capture window, and offers a Cancel button.
///
/// Carries **no instruction text** (global rule across calibration and all
/// four experiments): the participant is briefed verbally, and the dot's
/// colour carries the state — yellow while settling, green while the window
/// is actually capturing. A dot counter is kept, dimmed, because an operator
/// needs to know how far through the run they are; a failure message is kept
/// because a silent failure would otherwise look like a hang.
struct CalibrationOverlay: View {
    @ObservedObject var controller: CalibrationController
    let screenSize: CGSize
    let onCancel: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Calibration dot — yellow during dwell, green during capture.
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
                    .fill(isCapturing ? Color.green : Color.yellow)
                    .frame(width: 18, height: 18)
            }
            .position(center)

            // Dot counter only — no prompt text. Low-contrast and pinned to
            // the top edge so it doesn't compete with the dot for fixation.
            VStack(spacing: 6) {
                Text("\(controller.dotIndex + 1) / \(controller.targets.count)")
                    .font(.caption.monospaced())
                    .foregroundColor(.white.opacity(0.4))
                if case .failed(let message) = controller.phase {
                    Text(message)
                        .font(.caption)
                        .foregroundColor(.red)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 24)
                }
            }
            .frame(maxHeight: .infinity, alignment: .top)
            .padding(.top, 30)

            // Offset to 28% of the width rather than centred: the calibration
            // targets sit at 10% / 50% / 90% of both axes, so a centred button
            // at the bottom sits right on the bottom-centre target the
            // participant has to fixate.
            Button(action: onCancel) {
                Text("Cancel")
                    .font(.subheadline.weight(.medium))
                    .padding(.horizontal, 16)
                    .padding(.vertical, 7)
                    .background(Color.white.opacity(0.16))
                    .foregroundColor(.white)
                    .cornerRadius(9)
            }
            .position(x: screenSize.width * 0.28,
                      y: screenSize.height - 26)
        }
    }
}
