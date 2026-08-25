import SwiftUI

/// Full-screen overlay for Experiment 4 (prompted predictive communication).
///
/// Two phases with deliberately different screens:
///
///  * **Prompt** — the question, large and centred, spoken at the same time.
///    The grid is not drawn at all, so there is nothing to fixate and no dwell
///    can be building when the response phase opens.
///  * **Response** — the question text is gone and the 3×3 grid is live. Only
///    the answer composed so far is on screen, mirroring the spoken feedback.
///
/// This is how the codebase's "no on-screen instructions" rule (README) is
/// honoured here: nothing competes with the stimulus while gaze is scored. The
/// question has to be presented somehow, and presenting it before the grid
/// exists costs nothing measurable — whereas leaving it up during the response
/// would pull gaze to the top bar and add uncontrolled reading time to the
/// rate.
struct PredictiveTaskOverlay: View {
    @ObservedObject var controller: PredictiveTaskController
    let screenSize: CGSize
    let livePrediction: CGPoint?
    let onCancel: () -> Void
    let onFinish: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if case .failed(let why) = controller.phase {
                // A misauthored question set fails at start(); without this
                // the overlay would sit on a black screen with an empty grid
                // and the operator would have no idea why.
                failureView(why)
            } else if controller.phase == .prompt {
                promptView
            } else {
                gridView
                composedStrip
            }

            if controller.phase != .prompt, let p = livePrediction {
                gazeCursor.position(p).allowsHitTesting(false)
            }

            HStack(spacing: 12) {
                Button(action: onCancel) { controlLabel("Cancel") }
                Button(action: onFinish) { controlLabel("Finish") }
            }
            .frame(maxHeight: .infinity, alignment: .bottom)
            .padding(.bottom, 22)
        }
    }

    // MARK: - Prompt phase

    private var promptView: some View {
        VStack(spacing: 18) {
            Text(controller.currentQuestion?.text ?? "")
                .font(.system(size: 34, weight: .semibold, design: .rounded))
                .multilineTextAlignment(.center)
                .foregroundColor(.white)
                .padding(.horizontal, 28)
            Image(systemName: "speaker.wave.2.fill")
                .font(.title2)
                .foregroundColor(.white.opacity(0.45))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .transition(.opacity)
    }

    private func failureView(_ why: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.largeTitle)
                .foregroundColor(.yellow)
            Text(why)
                .font(.callout.monospaced())
                .multilineTextAlignment(.center)
                .foregroundColor(.yellow)
                .padding(.horizontal, 28)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Response phase

    private var gridView: some View {
        ForEach(controller.cells) { cell in
            cellView(for: cell)
                .frame(width: cell.rect.width - 6,
                       height: cell.rect.height - 6)
                .position(x: cell.rect.midX, y: cell.rect.midY)
        }
        // Keyed on the step so a refresh reads as a refresh rather than as
        // words silently mutating under the participant's fixation. The
        // controller's lockout covers the same window, so nothing is
        // selectable while this plays.
        .id(controller.gridStep)
        .transition(.opacity.combined(with: .scale(scale: 0.97)))
        .animation(.easeOut(duration: 0.18), value: controller.gridStep)
        .opacity(controller.isLockedOut ? 0.55 : 1.0)
    }

    /// The answer as composed so far. All selections read white, matching
    /// Experiment 3's strip: it mirrors what was picked and does not flag
    /// mis-picks. Colouring errors would put a correctness signal on screen
    /// that no real AAC device gives, and would cue corrections the
    /// participant would not otherwise have made.
    private var composedStrip: some View {
        VStack(spacing: 4) {
            HStack(spacing: 8) {
                ForEach(Array(controller.composed.enumerated()),
                        id: \.offset) { _, word in
                    Text(word)
                        .font(.title3.weight(.semibold))
                        .foregroundColor(.white)
                }
            }
            .frame(maxWidth: .infinity)
            .animation(.easeOut(duration: 0.15),
                       value: controller.composed.count)

            // Bare progress count, not an instruction.
            Text("\(controller.trialIndex + 1) / \(controller.questions.count)")
                .font(.caption2.monospaced())
                .foregroundColor(.white.opacity(0.4))
        }
        .frame(height: controller.sentenceBarHeight)
        .frame(maxHeight: .infinity, alignment: .top)
    }

    private var gazeCursor: some View {
        ZStack {
            Circle()
                .trim(from: 0, to: CGFloat(controller.dwellProgress))
                .stroke(Color.green,
                        style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .frame(width: 36, height: 36)
            Circle()
                .strokeBorder(Color.white.opacity(0.9), lineWidth: 2)
                .background(Circle().fill(Color.cyan.opacity(0.9)))
                .frame(width: 20, height: 20)
        }
    }

    @ViewBuilder
    private func cellView(for cell: PredictiveCell) -> some View {
        let isActive = controller.activeCellIndex == cell.index
        let wasSelected = controller.lastSelectedCell == cell.index
        ZStack {
            RoundedRectangle(cornerRadius: 10)
                .fill(fill(for: cell))
            content(for: cell).padding(6)
            RoundedRectangle(cornerRadius: 10)
                .stroke(isActive ? Color.green
                        : (wasSelected ? Color.blue : Color.white.opacity(0.25)),
                        lineWidth: isActive ? 4 : 2)
        }
    }

    /// Controls are visually distinct from words so they can be found without
    /// reading — they are in fixed positions all run and should become a
    /// motor habit rather than a search.
    private func fill(for cell: PredictiveCell) -> Color {
        switch cell.kind {
        case .word: return .white
        case .back: return Color(white: 0.32)
        case .done: return Color(red: 0.16, green: 0.42, blue: 0.24)
        case .blank: return Color(white: 0.10)
        }
    }

    @ViewBuilder
    private func content(for cell: PredictiveCell) -> some View {
        switch cell.kind {
        case .word:
            wordTile(cell)
        case .back:
            controlGlyph("delete.left", tint: .white)
        case .done:
            controlGlyph("checkmark", tint: .white)
        case .blank:
            EmptyView()
        }
    }

    /// The bundled word PNG when one exists (shared with Experiment 3's
    /// `word_<lowercase>.png` assets), otherwise the word as text. The
    /// predictor's vocabulary is much larger than the bundled image set, so
    /// unlike Experiment 3 the text path is the normal case, not a fallback
    /// for a missing asset.
    @ViewBuilder
    private func wordTile(_ cell: PredictiveCell) -> some View {
        if let ui = CommunicationTaskOverlay.loadImage(name: cell.imageName) {
            Image(uiImage: ui)
                .resizable()
                .scaledToFit()
        } else {
            Text(cell.word)
                .font(.system(size: 30, weight: .bold, design: .rounded))
                .foregroundColor(.black)
                .minimumScaleFactor(0.4)
                .lineLimit(1)
        }
    }

    private func controlGlyph(_ name: String, tint: Color) -> some View {
        Image(systemName: name)
            .font(.system(size: 34, weight: .semibold))
            .foregroundColor(tint)
    }

    private func controlLabel(_ title: String) -> some View {
        Text(title)
            .font(.body.weight(.medium))
            .padding(.horizontal, 22)
            .padding(.vertical, 9)
            .background(Color.white.opacity(0.16))
            .foregroundColor(.white)
            .cornerRadius(10)
    }
}
