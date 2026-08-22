import SwiftUI

/// Full-screen overlay shown while a `GridExperimentController` is running.
/// Draws a faint grid for context and fills the current target cell in red.
/// The red intensifies during the capture window so the user can see the
/// system is measuring (mirrors `AccuracyTestOverlay`).
struct GridExperimentOverlay: View {
    @ObservedObject var controller: GridExperimentController
    let screenSize: CGSize
    let onCancel: () -> Void

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Grid lines (faint, just for context).
            GridLines(rows: controller.gridSize.rows,
                      cols: controller.gridSize.cols,
                      size: screenSize)
                .stroke(Color.white.opacity(0.15), lineWidth: 0.5)

            // Target dot at the current cell's center (mirrors AccuracyTestOverlay).
            if let rect = controller.currentCellRect {
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
                .position(x: rect.midX, y: rect.midY)
            }

            // Trial counter only — no instruction text. The target dot's size
            // and white ring already signal when the capture window is open.
            VStack(spacing: 6) {
                Text("\(controller.trialIndex + 1) / \(controller.cellOrder.count)  ·  \(controller.gridSize.label)")
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
}

/// Simple grid-line `Shape` — `cols-1` vertical strokes and `rows-1`
/// horizontal strokes, just enough so the user can see the layout.
struct GridLines: Shape {
    let rows: Int
    let cols: Int
    let size: CGSize

    func path(in rect: CGRect) -> Path {
        var p = Path()
        guard rows > 0, cols > 0 else { return p }
        let cellW = rect.width  / CGFloat(cols)
        let cellH = rect.height / CGFloat(rows)
        for c in 1..<cols {
            let x = CGFloat(c) * cellW
            p.move(to: CGPoint(x: x, y: 0))
            p.addLine(to: CGPoint(x: x, y: rect.height))
        }
        for r in 1..<rows {
            let y = CGFloat(r) * cellH
            p.move(to: CGPoint(x: 0,         y: y))
            p.addLine(to: CGPoint(x: rect.width, y: y))
        }
        return p
    }
}
