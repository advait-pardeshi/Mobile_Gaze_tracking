import SwiftUI

/// Full-screen overlay for Experiment 3 (image-based communication task).
///
/// Layout: a composed-sentence strip along the top, the word-image grid in the
/// middle, controls at the bottom. The grid is inset from both bars by the
/// controller's layout so a fixation on either can never register as a word.
///
/// Per the global rule there are no instructions and **no prompt showing the
/// target sentence** — the participant is told the sentence to compose verbally
/// before the run. What is on screen is feedback only: the words selected so
/// far, mirroring the spoken audio.
struct CommunicationTaskOverlay: View {
    @ObservedObject var controller: CommunicationTaskController
    let screenSize: CGSize
    let livePrediction: CGPoint?
    let onCancel: () -> Void
    let onFinish: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            ForEach(controller.cells) { cell in
                cellView(for: cell)
                    .frame(width: cell.rect.width - 6,
                           height: cell.rect.height - 6)
                    .position(x: cell.rect.midX, y: cell.rect.midY)
            }

            sentenceStrip

            // Live gaze cursor — the participant needs it to aim at a word.
            // The dwell ring around it fills as the selection accumulates.
            if let p = livePrediction {
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
                .position(p)
                .allowsHitTesting(false)
            }

            HStack(spacing: 12) {
                Button(action: onCancel) {
                    controlLabel("Cancel")
                }
                Button(action: onFinish) {
                    controlLabel("Finish")
                }
            }
            .frame(maxHeight: .infinity, alignment: .bottom)
            .padding(.bottom, 22)
        }
    }

    /// The sentence as composed so far. All selections read white — the
    /// strip only mirrors what was picked, without flagging mis-picks.
    private var sentenceStrip: some View {
        VStack(spacing: 4) {
            HStack(spacing: 8) {
                ForEach(Array(controller.selections.enumerated()),
                        id: \.offset) { _, sel in
                    Text(sel.word)
                        .font(.title3.weight(.semibold))
                        .foregroundColor(.white)
                }
            }
            .frame(maxWidth: .infinity)
            .animation(.easeOut(duration: 0.15),
                       value: controller.selections.count)

            // Bare progress count, not an instruction.
            Text("\(controller.expectedIndex) / \(controller.wordSet.target.count)")
                .font(.caption2.monospaced())
                .foregroundColor(.white.opacity(0.4))
        }
        .frame(height: controller.sentenceBarHeight)
        .frame(maxHeight: .infinity, alignment: .top)
    }

    @ViewBuilder
    private func cellView(for cell: CommunicationCell) -> some View {
        let isActive = controller.activeCellIndex == cell.index
        let wasSelected = controller.lastSelectedCell == cell.index
        ZStack {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.white)
            wordImage(for: cell)
                .padding(6)
            RoundedRectangle(cornerRadius: 10)
                .stroke(isActive ? Color.green
                        : (wasSelected ? Color.blue : Color.white.opacity(0.25)),
                        lineWidth: isActive ? 4 : 2)
        }
    }

    /// The bundled word PNG, or a rendered text tile when the asset is
    /// missing — so the experiment still runs before the images are added.
    @ViewBuilder
    private func wordImage(for cell: CommunicationCell) -> some View {
        if let ui = Self.loadImage(name: cell.imageName) {
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

    private func controlLabel(_ title: String) -> some View {
        Text(title)
            .font(.body.weight(.medium))
            .padding(.horizontal, 22)
            .padding(.vertical, 9)
            .background(Color.white.opacity(0.16))
            .foregroundColor(.white)
            .cornerRadius(10)
    }

    /// xcodegen bundles the word PNGs flat at the resource root (it enumerates
    /// the directory), so look them up by basename.
    static func loadImage(name: String) -> UIImage? {
        if let url = Bundle.main.url(forResource: name, withExtension: "png"),
           let data = try? Data(contentsOf: url) {
            return UIImage(data: data)
        }
        return UIImage(named: name)
    }
}
