import SwiftUI
import UIKit
import simd

/// UIActivityViewController wrapper for handing the run bundle to AirDrop /
/// Files. Each results view keeps its own private copy (matching the existing
/// views in this project) so no shared symbol is needed.
private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

/// Results screen for a calibration validation run.
///
/// This is the gate before any experiment: it leads with a verdict, not with
/// raw numbers, because the decision the operator has to make is binary —
/// proceed, or re-calibrate.
struct CalibrationValidationResultsView: View {
    let result: CalibrationValidationResult
    let screenSize: CGSize
    /// Accept the calibration and return to the main screen.
    let onAccept: () -> Void
    /// Discard and start a fresh calibration.
    let onRecalibrate: () -> Void
    /// Re-run validation against the same calibration.
    let onRerun: () -> Void

    @State private var shareURL: URL?
    @State private var showShare = false
    @State private var exportError: String?

    var body: some View {
        ZStack {
            Color.black.opacity(0.93).ignoresSafeArea()

            errorVectorCanvas

            VStack(spacing: 14) {
                verdictBanner
                summaryBlock
            }
            .padding(.horizontal, 18)
            .padding(.vertical, 14)
            .background(Color.black.opacity(0.72))
            .cornerRadius(14)
            .frame(maxHeight: .infinity, alignment: .top)
            .padding(.top, 50)
            .padding(.horizontal, 12)

            VStack(spacing: 10) {
                if let err = exportError {
                    Text(err).font(.caption2).foregroundColor(.red)
                }
                HStack(spacing: 10) {
                    actionButton("Accept", color: .green, action: onAccept)
                    actionButton("Re-validate", color: .blue, action: onRerun)
                    actionButton("Re-calibrate", color: .orange,
                                 action: onRecalibrate)
                }
                Button(action: exportBundle) {
                    Text("Export run bundle")
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 18)
                        .padding(.vertical, 7)
                        .background(Color.white.opacity(0.16))
                        .foregroundColor(.white)
                        .cornerRadius(9)
                }
            }
            .frame(maxHeight: .infinity, alignment: .bottom)
            .padding(.bottom, 40)
            .sheet(isPresented: $showShare) {
                if let url = shareURL { ShareSheet(items: [url]) }
            }

            legend
        }
    }

    /// Each dot drawn where the validation showed it, with a yellow vector to
    /// the mean prediction. A consistent vector direction across all 9 dots
    /// means a residual global offset; vectors fanning outward mean the
    /// fitted distance `tz` is off.
    private var errorVectorCanvas: some View {
        Canvas { ctx, _ in
            let cx = screenSize.width * 0.5
            let cy = screenSize.height * 0.5
            for d in result.dots {
                let target = CGPoint(x: cx + CGFloat(d.target.x),
                                     y: cy + CGFloat(d.target.y))
                let targetRect = CGRect(x: target.x - 8, y: target.y - 8,
                                        width: 16, height: 16)
                if let s = d.scatter {
                    let mean = CGPoint(x: cx + CGFloat(s.mean.x),
                                       y: cy + CGFloat(s.mean.y))
                    // Precision circle: RMS scatter radius about the centroid.
                    if s.rms.isFinite, s.rms > 0 {
                        let r = CGFloat(s.rms)
                        ctx.stroke(Path(ellipseIn: CGRect(x: mean.x - r,
                                                          y: mean.y - r,
                                                          width: r * 2,
                                                          height: r * 2)),
                                   with: .color(.cyan.opacity(0.45)),
                                   lineWidth: 1)
                    }
                    var line = Path()
                    line.move(to: target)
                    line.addLine(to: mean)
                    ctx.stroke(line, with: .color(.yellow.opacity(0.9)),
                               lineWidth: 2)
                    ctx.fill(Path(ellipseIn: CGRect(x: mean.x - 5,
                                                    y: mean.y - 5,
                                                    width: 10, height: 10)),
                             with: .color(.yellow))
                    ctx.fill(Path(ellipseIn: targetRect), with: .color(.green))
                } else {
                    // Never confirmed — hollow grey marker.
                    ctx.stroke(Path(ellipseIn: targetRect),
                               with: .color(.white.opacity(0.4)), lineWidth: 1.5)
                }
            }
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }

    private var verdictBanner: some View {
        VStack(spacing: 4) {
            Text(result.verdict.label)
                .font(.headline.weight(.semibold))
                .foregroundColor(verdictColor)
                .multilineTextAlignment(.center)
            Text(result.verdict.advice)
                .font(.caption2)
                .foregroundColor(.white.opacity(0.75))
                .multilineTextAlignment(.center)
        }
    }

    private var verdictColor: Color {
        switch result.verdict {
        case .good:       return .green
        case .marginal:   return .yellow
        case .poor:       return .red
        case .incomplete: return .orange
        }
    }

    private var summaryBlock: some View {
        VStack(spacing: 8) {
            HStack(spacing: 18) {
                metric(label: "Mean err",
                       value: MetricFormat.degrees(result.meanErrorDegrees))
                metric(label: "Worst",
                       value: MetricFormat.degrees(result.worstErrorDegrees))
                metric(label: "Precision",
                       value: MetricFormat.degrees(result.meanPrecisionDegrees))
                metric(label: "Dots",
                       value: "\(result.confirmedCount)/\(result.dots.count)")
            }
            Divider().background(Color.white.opacity(0.3))
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text("dot").frame(width: 26, alignment: .leading)
                    Text("err pt").frame(width: 52, alignment: .trailing)
                    Text("err °").frame(width: 44, alignment: .trailing)
                    Text("rms").frame(width: 40, alignment: .trailing)
                    Text("n").frame(width: 40, alignment: .trailing)
                }
                .font(.system(.caption2, design: .monospaced))
                .foregroundColor(.white.opacity(0.5))
                ForEach(result.dots) { d in
                    dotRow(d)
                }
            }
        }
    }

    private func dotRow(_ d: ValidationDot) -> some View {
        let deg = VisualAngle.degrees(points: d.errorPoints,
                                      distancePoints: result.distancePoints)
        return HStack {
            Text("\(d.index + 1)")
                .frame(width: 26, alignment: .leading)
            Text(MetricFormat.number(d.errorPoints))
                .frame(width: 52, alignment: .trailing)
            Text(MetricFormat.number(deg, places: 2))
                .frame(width: 44, alignment: .trailing)
            Text(MetricFormat.number(d.precisionPoints))
                .frame(width: 40, alignment: .trailing)
            Text("\(d.sampleCount)")
                .frame(width: 40, alignment: .trailing)
        }
        .font(.system(.caption2, design: .monospaced))
        .foregroundColor(d.isConfirmed
                         ? .white.opacity(0.9) : .orange.opacity(0.8))
    }

    private var legend: some View {
        VStack(alignment: .leading, spacing: 4) {
            legendRow(color: .green,  label: "Dot position")
            legendRow(color: .yellow, label: "Mean prediction")
            legendRow(color: .cyan,   label: "RMS scatter")
        }
        .font(.system(.caption2, design: .monospaced))
        .foregroundColor(.white.opacity(0.85))
        .padding(8)
        .background(Color.black.opacity(0.55))
        .cornerRadius(8)
        .frame(maxWidth: .infinity, maxHeight: .infinity,
               alignment: .bottomLeading)
        .padding(.leading, 12)
        .padding(.bottom, 130)
    }

    private func legendRow(color: Color, label: String) -> some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(label)
        }
    }

    private func metric(label: String, value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.callout.weight(.semibold).monospacedDigit())
                .foregroundColor(.green)
            Text(label)
                .font(.caption2)
                .foregroundColor(.white.opacity(0.7))
        }
    }

    private func actionButton(_ title: String, color: Color,
                             action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.body.weight(.medium))
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(color.opacity(0.85))
                .foregroundColor(.white)
                .cornerRadius(10)
        }
    }

    private func exportBundle() {
        guard let url = result.runBundleURL,
              FileManager.default.fileExists(atPath: url.path) else {
            exportError = "No run bundle on disk for this run."
            return
        }
        shareURL = url
        exportError = nil
        showShare = true
    }
}
