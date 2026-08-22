import SwiftUI
import simd

/// Full-screen overlay for the calibration validation run.
///
/// All 9 dots are drawn at once and the participant works through them in any
/// order. Per the global "no on-screen instructions" rule there is no prompt
/// text: state is carried entirely by dot appearance.
///
///   * unconfirmed → hollow white ring with a white centre dot
///   * settling    → amber centre, with a ring that fills as dwell accumulates
///   * confirmed   → solid green with a tick
struct CalibrationValidationOverlay: View {
    @ObservedObject var controller: CalibrationValidationController
    let screenSize: CGSize
    /// Live Kalman-smoothed prediction, absolute screen points — the same
    /// point the controller scores.
    let livePrediction: CGPoint?
    let onCancel: () -> Void
    let onFinish: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            ForEach(controller.dots) { dot in
                dotView(for: dot)
                    .position(controller.absoluteCenter(of: dot))
            }

            // Live gaze cursor so the participant can see where the system
            // thinks they're looking — the only feedback channel left once
            // instruction text is gone.
            if let p = livePrediction {
                Circle()
                    .strokeBorder(Color.white.opacity(0.85), lineWidth: 2)
                    .background(Circle().fill(Color.cyan.opacity(0.85)))
                    .frame(width: 22, height: 22)
                    .position(p)
                    .allowsHitTesting(false)
            }

            // Progress is conveyed as a bare count of confirmed dots, not as
            // an instruction. Kept small and low-contrast so it doesn't pull
            // fixation away from the dots.
            Text("\(controller.dots.filter(\.isConfirmed).count) / \(controller.dots.count)")
                .font(.caption.monospaced())
                .foregroundColor(.white.opacity(0.45))
                .frame(maxHeight: .infinity, alignment: .top)
                .padding(.top, 24)

            // Controls are placed BETWEEN the dot columns, hard against the
            // bottom edge. The 9 dots sit at 10% / 50% / 90% of both axes, so
            // a centred button row at the bottom lands directly on top of the
            // bottom-centre dot — which the participant has to fixate. Putting
            // them at 28% / 72% of the width clears every dot column
            // horizontally, independent of screen size, and the bottom
            // alignment clears the bottom row vertically.
            controlButton("Cancel", action: onCancel)
                .position(x: screenSize.width * 0.28,
                          y: screenSize.height - 26)
            // Escape hatch: score what has been confirmed rather than waiting
            // out the 2-minute timeout on a dot the participant can't hold.
            controlButton("Finish", action: onFinish)
                .position(x: screenSize.width * 0.72,
                          y: screenSize.height - 26)
        }
    }

    private func controlButton(_ title: String,
                               action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.weight(.medium))
                .padding(.horizontal, 16)
                .padding(.vertical, 7)
                .background(Color.white.opacity(0.16))
                .foregroundColor(.white)
                .cornerRadius(9)
        }
    }

    @ViewBuilder
    private func dotView(for dot: ValidationDot) -> some View {
        let isActive = controller.activeDotIndex == dot.index
        ZStack {
            if dot.isConfirmed {
                Circle()
                    .fill(Color.green)
                    .frame(width: 30, height: 30)
                Image(systemName: "checkmark")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundColor(.black)
            } else {
                Circle()
                    .stroke(Color.white.opacity(isActive ? 0.5 : 0.3), lineWidth: 2)
                    .frame(width: 46, height: 46)
                // Dwell ring — fills clockwise from 12 o'clock while the
                // gaze holds this dot.
                if isActive {
                    Circle()
                        .trim(from: 0, to: CGFloat(controller.dwellProgress))
                        .stroke(Color.green,
                                style: StrokeStyle(lineWidth: 4, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                        .frame(width: 46, height: 46)
                }
                Circle()
                    .fill(isActive ? Color.orange : Color.white)
                    .frame(width: 18, height: 18)
            }
        }
        .frame(width: 52, height: 52)
    }
}
