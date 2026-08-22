import SwiftUI

/// Full-screen overlay for Experiment 2 (fixation stability, 4×4 grid).
///
/// Deliberately sparse: the grid of boxes, and one red circle in the cell
/// currently being held. No instruction text (global rule), and — unlike the
/// other experiments — **no live gaze cursor**. A moving cursor is a moving
/// stimulus: the participant tracks it instead of the circle, which is exactly
/// the drift this experiment is trying to measure.
///
/// Only the active cell is drawn brightly. The remaining cells stay very faint
/// so the participant has no competing high-contrast edge to saccade toward
/// mid-fixation, while the grid structure is still visible.
struct FixationStabilityOverlay: View {
    @ObservedObject var controller: FixationStabilityController
    let screenSize: CGSize
    let onCancel: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Faint grid so the layout reads as a 4×4 field.
            GridLines(rows: controller.rows,
                      cols: controller.cols,
                      size: screenSize)
                .stroke(Color.white.opacity(0.10), lineWidth: 0.5)

            // Active cell box.
            let box = controller.boxRect
            if box.width > 0 {
                Rectangle()
                    .stroke(Color.white.opacity(0.45), lineWidth: 2)
                    .frame(width: box.width - 4, height: box.height - 4)
                    .position(x: box.midX, y: box.midY)
            }

            // Capture-progress ring, sized to sit just outside the target so
            // its motion stays in peripheral vision rather than on the point
            // being fixated.
            if controller.phase == .capturing {
                Circle()
                    .trim(from: 0, to: CGFloat(controller.phaseProgress))
                    .stroke(Color.white.opacity(0.28),
                            style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                    .frame(width: 54, height: 54)
                    .position(controller.circleCenter)
            }

            // The fixation target. Slightly larger during the unscored settle
            // phase, so the participant has something to acquire before
            // measurement starts, then it steps down to its final size — a
            // static target for the whole scored window.
            Circle()
                .fill(Color.red)
                .frame(width: controller.phase == .settling ? 24 : 18,
                       height: controller.phase == .settling ? 24 : 18)
                .position(controller.circleCenter)

            // Bare trial counter — progress, not instruction.
            VStack(spacing: 6) {
                Text("\(controller.trialIndex + 1) / \(controller.trialCount)")
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
            .padding(.top, 12)

            // Horizontally centred is the SAFE position on an even-column
            // grid: 4 columns put the circles at 12.5% / 37.5% / 62.5% / 87.5%
            // of the width, so 50% falls exactly on a grid line, between two
            // circle columns. Pinning to the bottom edge then clears the
            // bottom row's circles (at 87.5% of the height) vertically too.
            Button(action: onCancel) {
                Text("Cancel")
                    .font(.subheadline.weight(.medium))
                    .padding(.horizontal, 16)
                    .padding(.vertical, 6)
                    .background(Color.white.opacity(0.16))
                    .foregroundColor(.white)
                    .cornerRadius(9)
            }
            .position(x: screenSize.width * 0.5,
                      y: screenSize.height - 16)
        }
    }
}
