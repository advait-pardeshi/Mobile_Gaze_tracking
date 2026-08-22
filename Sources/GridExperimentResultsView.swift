import SwiftUI
import UIKit

/// Results screen for one grid-focus experiment run.
/// Top:    headline accuracy + mean error.
/// Middle: grid heatmap — green = hit, red = miss; a small dot inside each
///         cell shows where the mean predicted gaze landed.
/// Bottom: Done / Run Again / Export CSV.
struct GridExperimentResultsView: View {
    let result: GridExperimentResult
    let screenSize: CGSize
    let onDismiss: () -> Void
    let onRerun: () -> Void

    @State private var shareURL: URL?
    @State private var showShare = false
    @State private var exportError: String?

    var body: some View {
        ZStack {
            Color.black.opacity(0.93).ignoresSafeArea()

            VStack(spacing: 6) {
                Text(result.gridSize.label)
                    .font(.title2.weight(.semibold))
                    .foregroundColor(.white)
                Text(String(format: "Accuracy: %.1f%%  (%d / %d hit)",
                            result.accuracy * 100.0,
                            result.hitCount, result.trials.count))
                    .font(.title3.weight(.semibold))
                    .foregroundColor(.green)
                Text("Mean error: "
                     + MetricFormat.pointsAndDegrees(result.meanErrorPoints,
                                                     result.meanErrorDegrees))
                    .font(.caption.monospaced())
                    .foregroundColor(.white.opacity(0.75))
                // The error budget for this resolution: half a cell. Mean
                // error above this means the grid is finer than the tracker
                // can resolve, which is the finding Experiment 1 looks for.
                Text(String(format: "Cell %.0f×%.0f pt   tolerance ±%@",
                            result.cellSize.width,
                            result.cellSize.height,
                            MetricFormat.degrees(result.cellToleranceDegrees)))
                    .font(.caption2.monospaced())
                    .foregroundColor(.white.opacity(0.5))
            }
            .frame(maxHeight: .infinity, alignment: .top)
            .padding(.top, 60)

            // Heatmap, scaled to ~70% of the screen.
            GeometryReader { geo in
                let pad: CGFloat = 24
                let avail = CGSize(
                    width:  geo.size.width  - 2 * pad,
                    height: geo.size.height * 0.65
                )
                let scale = min(avail.width  / screenSize.width,
                                avail.height / screenSize.height)
                let displaySize = CGSize(
                    width:  screenSize.width  * scale,
                    height: screenSize.height * scale
                )
                ZStack {
                    // Cell fills (green for hits, red for misses).
                    ForEach(result.trials, id: \.order) { t in
                        Rectangle()
                            .fill((t.isHit ? Color.green : Color.red).opacity(0.45))
                            .frame(width: t.rect.width * scale,
                                   height: t.rect.height * scale)
                            .position(x: t.rect.midX * scale,
                                      y: t.rect.midY * scale)
                    }
                    // Grid lines.
                    GridLines(rows: result.gridSize.rows,
                              cols: result.gridSize.cols,
                              size: displaySize)
                        .stroke(Color.white.opacity(0.35), lineWidth: 0.5)
                        .frame(width: displaySize.width, height: displaySize.height)
                    // Predicted-gaze dots, one per trial.
                    ForEach(result.trials, id: \.order) { t in
                        if t.meanPredictionAbs.x.isFinite,
                           t.meanPredictionAbs.y.isFinite {
                            Circle()
                                .fill(t.isHit ? Color.white : Color.yellow)
                                .frame(width: 6, height: 6)
                                .position(x: t.meanPredictionAbs.x * scale,
                                          y: t.meanPredictionAbs.y * scale)
                        }
                    }
                }
                .frame(width: displaySize.width, height: displaySize.height)
                .position(x: geo.size.width * 0.5, y: geo.size.height * 0.5)
            }

            VStack(spacing: 10) {
                if let e = exportError {
                    Text(e)
                        .font(.caption2)
                        .foregroundColor(.red)
                }
                HStack(spacing: 14) {
                    Button(action: onDismiss) {
                        Text("Done")
                            .font(.body.weight(.medium))
                            .padding(.horizontal, 22)
                            .padding(.vertical, 10)
                            .background(Color.white.opacity(0.18))
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    Button(action: exportCSV) {
                        Text("Export CSV")
                            .font(.body.weight(.medium))
                            .padding(.horizontal, 22)
                            .padding(.vertical, 10)
                            .background(Color.green.opacity(0.85))
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    if result.fineTuneBundleURL != nil {
                        Button(action: exportBundle) {
                            Text("Export Bundle")
                                .font(.body.weight(.medium))
                                .padding(.horizontal, 22)
                                .padding(.vertical, 10)
                                .background(Color.orange.opacity(0.9))
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                    if result.runBundleURL != nil {
                        Button(action: exportRunBundle) {
                            Text("Export Run")
                                .font(.body.weight(.medium))
                                .padding(.horizontal, 22)
                                .padding(.vertical, 10)
                                .background(Color.purple.opacity(0.85))
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                    Button(action: onRerun) {
                        Text("Run Again")
                            .font(.body.weight(.medium))
                            .padding(.horizontal, 22)
                            .padding(.vertical, 10)
                            .background(Color.blue.opacity(0.85))
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                }
            }
            .frame(maxHeight: .infinity, alignment: .bottom)
            .padding(.bottom, 50)
            .sheet(isPresented: $showShare) {
                if let url = shareURL {
                    GridShareSheet(items: [url])
                }
            }
        }
    }

    private func exportCSV() {
        let url = GridExperimentResult.masterLogURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            exportError = "No log file yet — finish a run first."
            return
        }
        shareURL = url
        exportError = nil
        showShare = true
    }

    private func exportRunBundle() {
        guard let url = result.runBundleURL,
              FileManager.default.fileExists(atPath: url.path) else {
            exportError = "Run bundle missing on disk."
            return
        }
        shareURL = url
        exportError = nil
        showShare = true
    }

    private func exportBundle() {
        guard let url = result.fineTuneBundleURL,
              FileManager.default.fileExists(atPath: url.path) else {
            exportError = "Fine-tune bundle missing on disk."
            return
        }
        shareURL = url
        exportError = nil
        showShare = true
    }
}

/// `UIActivityViewController` bridge for share sheet (kept distinct from
/// the accuracy view's wrapper so neither file has a duplicate symbol).
private struct GridShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
