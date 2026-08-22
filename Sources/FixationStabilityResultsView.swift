import SwiftUI
import UIKit
import simd

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

/// Results screen for Experiment 2 (fixation stability, 4×4 grid).
///
/// The map is the point of this screen. Mean deviation and RMS collapse 16
/// fixations into single numbers, and two runs with identical aggregates can
/// have completely different spatial structure — uniform noise everywhere
/// versus a tracker that is tight in the centre and falls apart in the
/// corners. Only the per-cell view distinguishes them, and that distinction is
/// what decides whether a fine grid is usable across the whole screen or just
/// in the middle.
struct FixationStabilityResultsView: View {
    let result: FixationStabilityResult
    let screenSize: CGSize
    let onDismiss: () -> Void
    let onRerun: () -> Void

    @State private var shareURL: URL?
    @State private var showShare = false
    @State private var exportError: String?

    var body: some View {
        ZStack {
            Color.black.opacity(0.95).ignoresSafeArea()

            ScrollView {
                VStack(spacing: 14) {
                    header
                    metricsBlock
                    mapBlock
                    perCellTable
                }
                .padding(.horizontal, 16)
                .padding(.top, 50)
                .padding(.bottom, 120)
            }

            VStack(spacing: 8) {
                if let err = exportError {
                    Text(err).font(.caption2).foregroundColor(.red)
                }
                HStack(spacing: 12) {
                    button("Done", color: .white.opacity(0.18), action: onDismiss)
                    button("Export", color: .green.opacity(0.85),
                           action: exportBundle)
                    button("Run Again", color: .blue.opacity(0.85),
                           action: onRerun)
                }
            }
            .frame(maxHeight: .infinity, alignment: .bottom)
            .padding(.bottom, 34)
            .background(
                // Scrim so scrolled content passes *behind* the bar instead of
                // interleaving with the button labels.
                LinearGradient(colors: [.black.opacity(0), .black.opacity(0.92),
                                        .black.opacity(0.98)],
                               startPoint: .top, endPoint: .bottom)
                    .frame(height: 130)
                    .allowsHitTesting(false)
                    .frame(maxHeight: .infinity, alignment: .bottom)
                    .ignoresSafeArea()
            )
            .sheet(isPresented: $showShare) {
                if let url = shareURL { ShareSheet(items: [url]) }
            }
        }
    }

    private var header: some View {
        VStack(spacing: 3) {
            Text("Fixation Stability")
                .font(.title3.weight(.semibold))
                .foregroundColor(.white)
            Text("\(result.rows)×\(result.cols) · \(result.scoredCellCount)/\(result.cells.count) cells scored")
                .font(.caption2.monospaced())
                .foregroundColor(.white.opacity(0.55))
        }
    }

    private var metricsBlock: some View {
        VStack(spacing: 10) {
            HStack(spacing: 14) {
                metric("Mean dev",
                       MetricFormat.degrees(result.meanDeviationDegrees))
                metric("RMS", MetricFormat.degrees(result.rmsDegrees))
                metric("Worst cell",
                       MetricFormat.degrees(result.worstRmsDegrees))
                metric("Bias", MetricFormat.degrees(result.biasDegrees))
            }
            Divider().background(Color.white.opacity(0.3))
            VStack(alignment: .leading, spacing: 3) {
                row("Mean deviation",
                    MetricFormat.pointsAndDegrees(result.meanDeviationPoints,
                                                  result.meanDeviationDegrees))
                row("SD (radial)",
                    MetricFormat.pointsAndDegrees(result.standardDeviationPoints,
                                                  result.standardDeviationDegrees))
                row("RMS (precision)",
                    MetricFormat.pointsAndDegrees(result.rmsPoints,
                                                  result.rmsDegrees))
                row("Bias (centroid→circle)",
                    MetricFormat.pointsAndDegrees(result.biasPoints,
                                                  result.biasDegrees))
                row("SD x / y",
                    "\(MetricFormat.number(result.sdX)) / \(MetricFormat.points(result.sdY))")
                row("Inside own cell",
                    MetricFormat.percent(result.containment, places: 1))
                row("Samples",
                    String(format: "%d over %d×%.0fs  (%@ Hz)",
                           result.totalSampleCount,
                           result.scoredCellCount,
                           result.captureDuration,
                           MetricFormat.number(result.sampleRateHz)))
            }
        }
        .padding(12)
        .background(Color.white.opacity(0.07))
        .cornerRadius(12)
    }

    /// Scaled replica of the screen: every cell shaded by its precision, with
    /// each cell's samples, centroid and circle centre drawn in place.
    private var mapBlock: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("PER-CELL PRECISION")
                .font(.caption2.monospaced())
                .foregroundColor(.white.opacity(0.45))
            GeometryReader { geo in
                let scale = min(geo.size.width / screenSize.width,
                                geo.size.height / screenSize.height)
                let w = screenSize.width * scale
                let h = screenSize.height * scale
                ZStack {
                    ForEach(result.cells) { cell in
                        Rectangle()
                            .fill(shade(for: cell))
                            .frame(width: cell.rect.width * scale,
                                   height: cell.rect.height * scale)
                            .position(x: cell.rect.midX * scale,
                                      y: cell.rect.midY * scale)
                    }
                    GridLines(rows: result.rows, cols: result.cols,
                              size: CGSize(width: w, height: h))
                        .stroke(Color.white.opacity(0.3), lineWidth: 0.5)
                    scatterCanvas(scale: scale)
                }
                .frame(width: w, height: h)
                .position(x: geo.size.width * 0.5, y: geo.size.height * 0.5)
            }
            .frame(height: 320)
            legend
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.white.opacity(0.07))
        .cornerRadius(12)
    }

    private func scatterCanvas(scale: CGFloat) -> some View {
        Canvas { ctx, _ in
            let cx = screenSize.width * 0.5 * scale
            let cy = screenSize.height * 0.5 * scale
            for cell in result.cells {
                for s in cell.samples {
                    let p = CGPoint(x: cx + CGFloat(s.prediction.x) * scale,
                                    y: cy + CGFloat(s.prediction.y) * scale)
                    ctx.fill(Path(ellipseIn: CGRect(x: p.x - 0.8, y: p.y - 0.8,
                                                    width: 1.6, height: 1.6)),
                             with: .color(.cyan.opacity(0.5)))
                }
                let target = CGPoint(x: cx + CGFloat(cell.target.x) * scale,
                                     y: cy + CGFloat(cell.target.y) * scale)
                if let sc = cell.scatter {
                    let mean = CGPoint(x: cx + CGFloat(sc.mean.x) * scale,
                                       y: cy + CGFloat(sc.mean.y) * scale)
                    var line = Path()
                    line.move(to: target)
                    line.addLine(to: mean)
                    ctx.stroke(line, with: .color(.yellow.opacity(0.85)),
                               lineWidth: 1)
                    ctx.fill(Path(ellipseIn: CGRect(x: mean.x - 2.5,
                                                    y: mean.y - 2.5,
                                                    width: 5, height: 5)),
                             with: .color(.yellow))
                }
                ctx.fill(Path(ellipseIn: CGRect(x: target.x - 3,
                                                y: target.y - 3,
                                                width: 6, height: 6)),
                         with: .color(.red))
            }
        }
    }

    /// Green (tight) → red (loose), scaled against the worst cell in this run
    /// so the map always spans its own range rather than a fixed threshold
    /// that could render every cell the same colour.
    private func shade(for cell: FixationCell) -> Color {
        guard cell.rmsPoints.isFinite else {
            return Color.white.opacity(0.04)
        }
        let worst = result.cells
            .map(\.rmsPoints).filter { $0.isFinite }.max() ?? cell.rmsPoints
        guard worst > 0 else { return Color.green.opacity(0.28) }
        let f = min(1.0, max(0.0, cell.rmsPoints / worst))
        return Color(red: f, green: 1.0 - f, blue: 0.25).opacity(0.28)
    }

    private var perCellTable: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("PER-CELL")
                .font(.caption2.monospaced())
                .foregroundColor(.white.opacity(0.45))
            HStack {
                Text("r,c").frame(width: 34, alignment: .leading)
                Text("dev °").frame(width: 46, alignment: .trailing)
                Text("rms °").frame(width: 46, alignment: .trailing)
                Text("bias °").frame(width: 46, alignment: .trailing)
                Text("in %").frame(width: 44, alignment: .trailing)
                Text("n").frame(width: 38, alignment: .trailing)
            }
            .font(.system(.caption2, design: .monospaced))
            .foregroundColor(.white.opacity(0.45))
            ForEach(result.cells) { c in
                cellRow(c)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.white.opacity(0.07))
        .cornerRadius(12)
    }

    private func cellRow(_ c: FixationCell) -> some View {
        let d = result.distancePoints
        let scored = !c.samples.isEmpty
        return HStack {
            Text("\(c.row),\(c.col)").frame(width: 34, alignment: .leading)
            Text(MetricFormat.number(
                VisualAngle.degrees(points: c.meanDeviationPoints,
                                    distancePoints: d), places: 2))
                .frame(width: 46, alignment: .trailing)
            Text(MetricFormat.number(
                VisualAngle.degrees(points: c.rmsPoints, distancePoints: d),
                places: 2))
                .frame(width: 46, alignment: .trailing)
            Text(MetricFormat.number(
                VisualAngle.degrees(points: c.biasPoints, distancePoints: d),
                places: 2))
                .frame(width: 46, alignment: .trailing)
            Text(MetricFormat.number(
                c.containment(screenSize: screenSize) * 100.0, places: 0))
                .frame(width: 44, alignment: .trailing)
            Text("\(c.sampleCount)").frame(width: 38, alignment: .trailing)
        }
        .font(.system(.caption2, design: .monospaced))
        .foregroundColor(scored ? .white.opacity(0.9) : .orange.opacity(0.8))
    }

    private var legend: some View {
        HStack(spacing: 12) {
            legendRow(color: .red, label: "Circle")
            legendRow(color: .yellow, label: "Centroid")
            legendRow(color: .cyan, label: "Samples")
        }
        .font(.system(.caption2, design: .monospaced))
        .foregroundColor(.white.opacity(0.75))
    }

    private func legendRow(color: Color, label: String) -> some View {
        HStack(spacing: 5) {
            Circle().fill(color).frame(width: 7, height: 7)
            Text(label)
        }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label).foregroundColor(.white.opacity(0.65))
            Spacer(minLength: 10)
            Text(value).foregroundColor(.white.opacity(0.95))
        }
        .font(.system(.caption2, design: .monospaced))
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.callout.weight(.semibold).monospacedDigit())
                .foregroundColor(.green)
            Text(label)
                .font(.caption2)
                .foregroundColor(.white.opacity(0.7))
        }
    }

    private func button(_ title: String, color: Color,
                        action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.body.weight(.medium))
                .padding(.horizontal, 18)
                .padding(.vertical, 10)
                .background(color)
                .foregroundColor(.white)
                .cornerRadius(10)
        }
    }

    private func exportBundle() {
        // Prefer this run's self-describing bundle; fall back to the
        // cumulative master log, which is always written on completion.
        if let url = result.runBundleURL,
           FileManager.default.fileExists(atPath: url.path) {
            shareURL = url
            exportError = nil
            showShare = true
            return
        }
        let url = FixationStabilityResult.masterLogURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            exportError = "No log file yet — finish a run first."
            return
        }
        shareURL = url
        exportError = nil
        showShare = true
    }
}
