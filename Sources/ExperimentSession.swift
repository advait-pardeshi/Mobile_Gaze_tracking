import Foundation
import Combine

/// The runs completed since the app was launched, and a one-tap export of
/// them.
///
/// **Why this exists.** Every run already writes a self-describing bundle to
/// `Documents/experiment_runs/`, and every experiment also appends to a
/// cumulative master log. Neither is what an operator wants at the end of a
/// data-collection sitting: the per-run zip is one grid out of the several
/// they just ran, and the master log is *every run the phone has ever done*,
/// including yesterday's and another participant's. Sharing either one means
/// hand-filtering afterwards.
///
/// So the unit of export is the **session**: the runs of one experiment,
/// completed since this launch of the app. Run 3×3, then 4×4, then 5×4, tap
/// Share, and the zip holds exactly those three and nothing else.
///
/// **Session identity is process lifetime, deliberately.** There is no
/// persistence here and no "clear" button to forget: relaunching the app is
/// the reset, which is the one gesture an operator cannot forget to perform
/// between participants. Runs from a previous launch stay on disk and stay in
/// the master log — nothing is destroyed — they simply are not in this
/// session's export.
@MainActor
final class ExperimentSession: ObservableObject {

    static let shared = ExperimentSession()

    /// One completed run: where its bundle directory is, and enough identity
    /// to build the manifest without re-parsing the experiment's types.
    struct Run {
        let experiment: String
        let label: String
        let directory: URL
        let finishedAt: Date
    }

    /// Stamp for this launch. Names the export so two sessions' archives
    /// can't collide in the share destination.
    let sessionStamp: String
    let startedAt = Date()

    /// Completed runs, in the order they were run. Published so a button can
    /// show the count and enable itself the moment the first run lands.
    @Published private(set) var runs: [Run] = []

    private init() {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd_HHmmss"
        sessionStamp = f.string(from: Date())
    }

    // MARK: - Recording

    /// Called by each experiment as it finishes a run. `directory` is the
    /// bundle directory (not the zip): the export copies its contents, so a
    /// failed zip earlier in the chain doesn't cost the run its place here.
    func record(experiment: String, label: String, directory: URL) {
        runs.append(Run(experiment: experiment,
                        label: label,
                        directory: directory,
                        finishedAt: Date()))
    }

    func runs(of experiment: String) -> [Run] {
        runs.filter { $0.experiment == experiment }
    }

    func runCount(of experiment: String) -> Int {
        runs(of: experiment).count
    }

    // MARK: - Export

    enum ExportError: LocalizedError {
        case noRuns(String)
        case zipFailed

        var errorDescription: String? {
            switch self {
            case .noRuns(let exp):
                return "No \(exp) runs in this session yet."
            case .zipFailed:
                return "Could not build the archive."
            }
        }
    }

    /// Every experiment that has at least one run in this session, in the
    /// order they were first run. Drives the "everything" export.
    var experimentsPresent: [String] {
        var seen = Set<String>()
        return runs.compactMap { seen.insert($0.experiment).inserted ? $0.experiment : nil }
    }

    /// Build one zip holding **every run of every experiment** from this
    /// session, grouped by experiment.
    ///
    /// ```
    ///   session_<stamp>/
    ///     session.json
    ///     exp1/  trials_all_runs.csv, run1_3x3/, run2_4x4/ …
    ///     exp3/  trials_all_runs.csv, run1_communication/ …
    ///     exp4/  …
    /// ```
    /// Use this at the end of a participant's whole sitting;
    /// `export(experiment:)` is the one-experiment version.
    func exportAll() throws -> URL {
        guard !runs.isEmpty else { throw ExportError.noRuns("any experiment") }

        let name = "session_\(sessionStamp)"
        let root = Self.exportsRoot()
        let staging = root.appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.removeItem(at: staging)
        try FileManager.default.createDirectory(at: staging,
                                                withIntermediateDirectories: true)

        let iso = ISO8601DateFormatter()
        var groups: [String: Any] = [:]
        for exp in experimentsPresent {
            let sub = staging.appendingPathComponent(exp, isDirectory: true)
            try FileManager.default.createDirectory(at: sub,
                                                    withIntermediateDirectories: true)
            groups[exp] = try stage(experiment: exp, into: sub)
        }

        let manifest: [String: Any] = [
            "session_stamp": sessionStamp,
            "session_started_at": iso.string(from: startedAt),
            "exported_at": iso.string(from: Date()),
            "experiments": experimentsPresent,
            "run_count": runs.count,
            "schema_version": ExperimentRunLog.schemaVersion,
            "pipeline_tuning": PipelineTuning.describe(),
            "by_experiment": groups,
        ]
        let data = try JSONSerialization.data(withJSONObject: manifest,
                                              options: [.prettyPrinted, .sortedKeys])
        try data.write(to: staging.appendingPathComponent("session.json"),
                       options: .atomic)

        let zipURL = root.appendingPathComponent("\(name).zip")
        guard DirectoryZip.zip(directory: staging, to: zipURL,
                               tag: "ExperimentSession") else {
            throw ExportError.zipFailed
        }
        return zipURL
    }

    /// Build one zip holding every run of `experiment` from this session.
    ///
    /// Layout inside the archive:
    /// ```
    ///   <exp>_session_<stamp>/
    ///     session.json          identity, run list, pipeline config
    ///     trials_all_runs.csv   every run's trials.csv, concatenated
    ///     run1_<label>/         samples.csv, trials.csv, meta.json
    ///     run2_<label>/         …
    /// ```
    /// `trials_all_runs.csv` is the analysis-ready table — the same body the
    /// master log carries, but scoped to this session — so the common case
    /// (plot accuracy against grid size) needs no unzipping of the per-run
    /// directories at all. The full bundles ride along for the sample-level
    /// work.
    func export(experiment: String) throws -> URL {
        guard !runs(of: experiment).isEmpty else {
            throw ExportError.noRuns(experiment)
        }

        let name = "\(experiment)_session_\(sessionStamp)"
        let root = Self.exportsRoot()
        let staging = root.appendingPathComponent(name, isDirectory: true)

        // Rebuilt from scratch every time: a session grows as more runs
        // finish, and a stale directory would silently ship an export that
        // is missing the run the operator just did.
        try? FileManager.default.removeItem(at: staging)
        try FileManager.default.createDirectory(at: staging,
                                                withIntermediateDirectories: true)

        let iso = ISO8601DateFormatter()
        let group = try stage(experiment: experiment, into: staging)

        let manifest: [String: Any] = [
            "experiment": experiment,
            "session_stamp": sessionStamp,
            "session_started_at": iso.string(from: startedAt),
            "exported_at": iso.string(from: Date()),
            "run_count": runs(of: experiment).count,
            "schema_version": ExperimentRunLog.schemaVersion,
            "pipeline_tuning": PipelineTuning.describe(),
            "runs": group,
        ]
        let data = try JSONSerialization.data(withJSONObject: manifest,
                                              options: [.prettyPrinted,
                                                        .sortedKeys])
        try data.write(to: staging.appendingPathComponent("session.json"),
                       options: .atomic)

        let zipURL = root.appendingPathComponent("\(name).zip")
        guard DirectoryZip.zip(directory: staging, to: zipURL,
                               tag: "ExperimentSession") else {
            throw ExportError.zipFailed
        }
        return zipURL
    }

    /// Copy one experiment's runs into `dir` and write its
    /// `trials_all_runs.csv`. Returns the manifest entries for those runs.
    ///
    /// Shared by both exports so the single-experiment archive and the
    /// per-experiment folder inside the everything archive can never drift
    /// into holding different things.
    @discardableResult
    private func stage(experiment: String, into dir: URL) throws -> [[String: Any]] {
        let mine = runs(of: experiment)
        var combined: [String] = []
        var entries: [[String: Any]] = []
        let iso = ISO8601DateFormatter()

        for (i, run) in mine.enumerated() {
            let n = i + 1
            let dest = dir.appendingPathComponent("run\(n)_\(Self.slug(run.label))",
                                                  isDirectory: true)
            try? FileManager.default.copyItem(at: run.directory, to: dest)

            let trials = run.directory.appendingPathComponent("trials.csv")
            if let body = try? String(contentsOf: trials, encoding: .utf8) {
                combined.append("=== Run \(n) — \(run.label) "
                                + "— \(iso.string(from: run.finishedAt)) ===")
                combined.append(body)
                combined.append("")
            }

            var entry: [String: Any] = [
                "run": n,
                "label": run.label,
                "directory": dest.lastPathComponent,
                "finished_at": iso.string(from: run.finishedAt),
            ]
            // The run's own meta.json is the authority on its parameters;
            // copying it into the manifest saves the analysis a second read.
            let metaURL = run.directory.appendingPathComponent("meta.json")
            if let d = try? Data(contentsOf: metaURL),
               let obj = try? JSONSerialization.jsonObject(with: d),
               let dict = obj as? [String: Any] {
                entry["meta"] = dict
            }
            entries.append(entry)
        }

        try combined.joined(separator: "\n")
            .write(to: dir.appendingPathComponent("trials_all_runs.csv"),
                   atomically: true, encoding: .utf8)
        return entries
    }

    static func exportsRoot() -> URL {
        let url = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("session_exports", isDirectory: true)
        try? FileManager.default.createDirectory(at: url,
                                                 withIntermediateDirectories: true)
        return url
    }

    /// Filename-safe form of a run label — the presets carry "×" and spaces
    /// ("9×9 walking"), which are legal in a path but awkward in a shell.
    private static func slug(_ s: String) -> String {
        let mapped = s.replacingOccurrences(of: "×", with: "x")
            .lowercased()
            .map { $0.isLetter || $0.isNumber ? $0 : "_" }
        return String(mapped)
            .split(separator: "_", omittingEmptySubsequences: true)
            .joined(separator: "_")
    }
}
