import SwiftUI
import UIKit

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

/// Results screen for Experiment 3 (communication task).
///
/// Leads with the composed sentence against the target, because that — not an
/// error in degrees — is the outcome this experiment is about. The rate figures
/// below it are what would be quoted for an AAC use case.
struct CommunicationTaskResultsView: View {
    let result: CommunicationTaskResult
    let screenSize: CGSize
    let onDismiss: () -> Void
    let onRerun: () -> Void
    /// Replays the composed sentence aloud.
    let onReplay: () -> Void

    @State private var shareURL: URL?
    @State private var showShare = false
    @State private var exportError: String?

    var body: some View {
        ZStack {
            Color.black.opacity(0.95).ignoresSafeArea()

            ScrollView {
                VStack(spacing: 16) {
                    header
                    sentenceComparison
                    metricsRow
                    selectionList
                }
                .padding(.horizontal, 18)
                .padding(.top, 54)
                .padding(.bottom, 150)
            }

            VStack(spacing: 8) {
                if let err = exportError {
                    Text(err).font(.caption2).foregroundColor(.red)
                }
                HStack(spacing: 10) {
                    button("Done", color: .white.opacity(0.18), action: onDismiss)
                    button("Replay", color: .purple.opacity(0.85),
                           action: onReplay)
                    button("Export", color: .green.opacity(0.85),
                           action: exportBundle)
                    button("Again", color: .blue.opacity(0.85), action: onRerun)
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
            .padding(.horizontal, 10)
            .sheet(isPresented: $showShare) {
                if let url = shareURL { ShareSheet(items: [url]) }
            }
        }
    }

    private var header: some View {
        VStack(spacing: 4) {
            Text("Communication Task")
                .font(.title3.weight(.semibold))
                .foregroundColor(.white)
            Text(result.completed
                 ? "Sentence completed"
                 : "Sentence not completed")
                .font(.caption.weight(.medium))
                .foregroundColor(result.completed ? .green : .orange)
        }
    }

    private var sentenceComparison: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 3) {
                Text("TARGET")
                    .font(.caption2.monospaced())
                    .foregroundColor(.white.opacity(0.45))
                Text(result.wordSet.target.joined(separator: " "))
                    .font(.body.weight(.medium))
                    .foregroundColor(.white.opacity(0.9))
            }
            VStack(alignment: .leading, spacing: 3) {
                Text("COMPOSED")
                    .font(.caption2.monospaced())
                    .foregroundColor(.white.opacity(0.45))
                // Wrong picks stay visible in red rather than being filtered
                // out — the sequence the participant actually produced is the
                // result, not a cleaned-up version of it.
                HStack(spacing: 6) {
                    ForEach(Array(result.selections.enumerated()),
                            id: \.offset) { _, sel in
                        Text(sel.word)
                            .font(.body.weight(.medium))
                            .foregroundColor(sel.isCorrect ? .green : .red)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.white.opacity(0.07))
        .cornerRadius(12)
    }

    private var metricsRow: some View {
        VStack(spacing: 10) {
            HStack(spacing: 14) {
                metric("Correct",
                       "\(result.correctCount)/\(result.wordSet.target.count)")
                metric("Wrong picks", "\(result.incorrectCount)")
                metric("Sel. acc", MetricFormat.percent(result.selectionAccuracy))
                metric("Rate",
                       "\(MetricFormat.number(result.wordsPerMinute)) wpm")
            }
            VStack(alignment: .leading, spacing: 3) {
                row("Total time",
                    MetricFormat.seconds(result.durationSeconds, places: 1))
                row("Mean per selection",
                    MetricFormat.seconds(result.meanSecondsPerSelection))
                row("Mean per correct word",
                    MetricFormat.seconds(result.meanSecondsPerCorrectWord))
                row("Selections", "\(result.selections.count)")
            }
        }
        .padding(12)
        .background(Color.white.opacity(0.07))
        .cornerRadius(12)
    }

    private var selectionList: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("SELECTIONS")
                .font(.caption2.monospaced())
                .foregroundColor(.white.opacity(0.45))
            HStack {
                Text("#").frame(width: 22, alignment: .leading)
                Text("word").frame(width: 76, alignment: .leading)
                Text("ok").frame(width: 24, alignment: .center)
                Text("at s").frame(width: 52, alignment: .trailing)
                Text("Δ s").frame(width: 52, alignment: .trailing)
            }
            .font(.system(.caption2, design: .monospaced))
            .foregroundColor(.white.opacity(0.45))
            ForEach(Array(result.selections.enumerated()),
                    id: \.offset) { _, s in
                HStack {
                    Text("\(s.order)").frame(width: 22, alignment: .leading)
                    Text(s.word).frame(width: 76, alignment: .leading)
                    Text(s.isCorrect ? "✓" : "✗")
                        .frame(width: 24, alignment: .center)
                    Text(String(format: "%.2f", s.atSeconds))
                        .frame(width: 52, alignment: .trailing)
                    Text(String(format: "%.2f", s.sincePreviousSeconds))
                        .frame(width: 52, alignment: .trailing)
                }
                .font(.system(.caption2, design: .monospaced))
                .foregroundColor(s.isCorrect
                                 ? .white.opacity(0.9) : .red.opacity(0.9))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.white.opacity(0.07))
        .cornerRadius(12)
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
                .font(.subheadline.weight(.medium))
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(color)
                .foregroundColor(.white)
                .cornerRadius(10)
        }
    }

    private func exportBundle() {
        if let url = result.runBundleURL,
           FileManager.default.fileExists(atPath: url.path) {
            shareURL = url
            exportError = nil
            showShare = true
            return
        }
        let url = CommunicationTaskResult.masterLogURL()
        guard FileManager.default.fileExists(atPath: url.path) else {
            exportError = "No log file yet — finish a run first."
            return
        }
        shareURL = url
        exportError = nil
        showShare = true
    }
}
