import SwiftUI
import UIKit
import simd

/// Full-screen results screen shown after an accuracy-test run completes.
/// Top: numeric summary (mean error in points + degrees).
/// Center: visual overlay — each target with its measured error vector and
/// per-sample scatter. Bottom: dismiss / re-run buttons.
/// UIActivityViewController wrapper so we can hand the CSV file to AirDrop /
/// Mail / Files / Save to Files from a SwiftUI `.sheet(...)`.
private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

struct AccuracyResultsView: View {
    let result: AccuracyTestResult
    let screenSize: CGSize
    let onDismiss: () -> Void
    let onRerun: () -> Void

    @State private var shareURL: URL?
    @State private var showShare = false
    @State private var exportError: String?

    var body: some View {
        ZStack {
            Color.black.opacity(0.92).ignoresSafeArea()

            // Per-point overlay rendered in absolute (screen-relative) coords
            // so dots line up exactly where the test displayed them.
            Canvas { ctx, _ in
                let cx = screenSize.width  * 0.5
                let cy = screenSize.height * 0.5
                for p in result.points {
                    let target = CGPoint(
                        x: cx + CGFloat(p.target.x),
                        y: cy + CGFloat(p.target.y)
                    )
                    let mean = CGPoint(
                        x: cx + CGFloat(p.mean.x),
                        y: cy + CGFloat(p.mean.y)
                    )

                    // Faint sample scatter (raw predictions).
                    for s in p.predictions {
                        let pt = CGPoint(
                            x: cx + CGFloat(s.x),
                            y: cy + CGFloat(s.y)
                        )
                        let rect = CGRect(x: pt.x - 2, y: pt.y - 2,
                                          width: 4, height: 4)
                        ctx.fill(Path(ellipseIn: rect),
                                 with: .color(.cyan.opacity(0.35)))
                    }

                    // Target (red, ground truth).
                    let targetRect = CGRect(x: target.x - 9, y: target.y - 9,
                                            width: 18, height: 18)
                    ctx.fill(Path(ellipseIn: targetRect),
                             with: .color(.red))
                    ctx.stroke(Path(ellipseIn: targetRect),
                               with: .color(.white.opacity(0.5)), lineWidth: 1)

                    // Error vector: from target → mean predicted point.
                    if !p.predictions.isEmpty {
                        var line = Path()
                        line.move(to: target)
                        line.addLine(to: mean)
                        ctx.stroke(line,
                                   with: .color(.yellow.opacity(0.9)),
                                   lineWidth: 2)
                        let meanRect = CGRect(x: mean.x - 6, y: mean.y - 6,
                                              width: 12, height: 12)
                        ctx.fill(Path(ellipseIn: meanRect),
                                 with: .color(.yellow))
                    }
                }
            }
            .ignoresSafeArea()
            .allowsHitTesting(false)

            VStack(spacing: 16) {
                Text("Accuracy Test")
                    .font(.title2.weight(.semibold))
                    .foregroundColor(.white)
                summaryBlock
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            .background(Color.black.opacity(0.7))
            .cornerRadius(14)
            .frame(maxHeight: .infinity, alignment: .top)
            .padding(.top, 60)

            VStack(spacing: 10) {
                if let err = exportError {
                    Text(err)
                        .font(.caption2)
                        .foregroundColor(.red)
                }
                HStack(spacing: 14) {
                    Button(action: onDismiss) {
                        Text("Done")
                            .font(.body.weight(.medium))
                            .padding(.horizontal, 24)
                            .padding(.vertical, 10)
                            .background(Color.white.opacity(0.18))
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    Button(action: exportCSV) {
                        Text("Export CSV")
                            .font(.body.weight(.medium))
                            .padding(.horizontal, 24)
                            .padding(.vertical, 10)
                            .background(Color.green.opacity(0.85))
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    Button(action: onRerun) {
                        Text("Run Again")
                            .font(.body.weight(.medium))
                            .padding(.horizontal, 24)
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
                    ShareSheet(items: [url])
                }
            }

            // Legend in the corner.
            VStack(alignment: .leading, spacing: 4) {
                legendRow(color: .red,    label: "Target")
                legendRow(color: .yellow, label: "Mean prediction")
                legendRow(color: .cyan,   label: "Samples")
            }
            .font(.system(.caption2, design: .monospaced))
            .foregroundColor(.white.opacity(0.85))
            .padding(8)
            .background(Color.black.opacity(0.55))
            .cornerRadius(8)
            .frame(maxWidth: .infinity, maxHeight: .infinity,
                   alignment: .bottomLeading)
            .padding(.leading, 14)
            .padding(.bottom, 120)
        }
    }

    private var summaryBlock: some View {
        VStack(spacing: 8) {
            HStack(spacing: 24) {
                metric(label: "Mean error",
                       value: String(format: "%.1f pt",
                                     result.meanErrorPoints))
                metric(label: "Mean angle",
                       value: String(format: "%.2f°",
                                     result.meanErrorDegrees))
            }
            Divider().background(Color.white.opacity(0.3))
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(result.points.enumerated()), id: \.offset) { i, p in
                    HStack {
                        Text("P\(i + 1)")
                            .frame(width: 28, alignment: .leading)
                        Text(String(format: "err %5.1f pt",
                                    p.errorPoints))
                        Spacer(minLength: 8)
                        Text(String(format: "(%5.0f,%5.0f)",
                                    p.errorVector.x, p.errorVector.y))
                        Spacer(minLength: 8)
                        Text(String(format: "σ %4.1f",
                                    p.rmsScatter.isFinite ? p.rmsScatter : 0))
                        Spacer(minLength: 8)
                        Text("n=\(p.predictions.count)")
                    }
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundColor(.white.opacity(0.9))
                }
            }
        }
    }

    private func metric(label: String, value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.title3.weight(.semibold).monospacedDigit())
                .foregroundColor(.green)
            Text(label)
                .font(.caption2)
                .foregroundColor(.white.opacity(0.7))
        }
    }

    private func exportCSV() {
        // The current run was already appended to the master log when it
        // finished (see GazeViewModel). Share that cumulative file so the
        // user gets every run, not just the most recent.
        let url = AccuracyTestResult.masterLogURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            exportError = "No log file yet — finish a run first."
            return
        }
        shareURL = url
        exportError = nil
        showShare = true
    }

    private func legendRow(color: Color, label: String) -> some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label)
        }
    }
}
