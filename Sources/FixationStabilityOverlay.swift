import SwiftUI

/// Full-screen overlay for Experiment 2 (fixation stability, 4×4 grid).
///
/// Deliberately sparse: one red circle in the cell currently being held, on a
/// plain black field. No cell boundaries or grid lines are drawn, so there is
/// no high-contrast edge to saccade toward mid-fixation, and no instruction
/// text (global rule). The live gaze
/// cursor is drawn so the prediction can be watched during a run — note that
/// a moving cursor is also a moving stimulus, so a participant who follows it
/// instead of the red circle will inflate the very drift this experiment
/// measures. It is drawn dim and small for that reason.
struct FixationStabilityOverlay: View {
    @ObservedObject var controller: FixationStabilityController
    let screenSize: CGSize
    /// Smoothed on-screen prediction, or nil when no estimate this frame.
    /// Under `PipelineTuning.fixationScoredStream == .filtered` this is the
    /// same point the run is scored on, so the cursor shows the measurement.
    let livePrediction: CGPoint?
    let onCancel: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // No grid lines and no cell box: the red target is the only
            // stimulus, so nothing competes with it for fixation.

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

            // Live prediction cursor, drawn above the target so it stays
            // readable when it lands on the circle.
            if let p = livePrediction {
                Circle()
                    .strokeBorder(Color.white.opacity(0.55), lineWidth: 1.5)
                    .background(Circle().fill(Color.cyan.opacity(0.55)))
                    .frame(width: 18, height: 18)
                    .position(p)
                    .allowsHitTesting(false)
            }

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
