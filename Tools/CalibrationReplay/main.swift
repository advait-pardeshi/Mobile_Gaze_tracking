import Foundation
import CoreGraphics
import simd

// Replay harness: score `CalibrationModel.LocalCorrection.nearestNeighbour`
// against `.gaussianRBF` on recorded Experiment 1 (grid) run bundles.
//
// It compiles the SHIPPING `CalibrationModel.swift` and `ScreenMapper.swift`
// (see ../replay_calibration.sh) rather than reimplementing them, so what is
// scored here is what runs on the phone.
//
// ---------------------------------------------------------------------------
// What the data allows, and what it does not
//
// A grid run bundle records, per frame, a gaze vector in camera coords and the
// cell the participant was told to fixate. That is the same (gaze, target)
// shape a calibration session has, so a calibration can be SYNTHESISED from it
// — but the original calibration that produced the run is NOT in the bundle
// (only its output, `pred_x/pred_y`). So this harness does not replay the
// participant's real fit; it builds a fresh one from their grid fixations and
// asks which blending rule generalises better across it. That is the question
// the branch is about, but it is not the same as "what would have happened in
// that session".
//
// Two evaluations are reported:
//
//   HELD-OUT  (headline) — per cell, the first `--fit-fraction` of that cell's
//     capture frames build the calibration and the rest are scored. Every one
//     of the nine anchors keeps training data, so neither variant is handed an
//     empty anchor, and no scored frame contributed to the fit.
//
//   IN-SAMPLE (reference) — fit on all frames, score all frames. Optimistic,
//     reported only so the gap between the two is visible.
//
// Frames inside one 2 s fixation are strongly autocorrelated, so a per-frame
// significance test would badly overstate its case. The headline paired
// statistics are therefore computed over PER-CELL mean errors, which is the
// level at which the samples are close to independent.
// ---------------------------------------------------------------------------

// MARK: - CSV / bundle loading

struct Frame {
    let cellIdx: Int
    let gazeCam: simd_double3
    /// Cell centre, screen-centre-relative points.
    let targetRel: simd_double2
    let order: Int
}

struct Run {
    let name: String
    let screenSize: CGSize
    let frames: [Frame]
    let rows: Int
    let cols: Int
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data(("error: " + msg + "\n").utf8))
    exit(1)
}

func findBundles(_ roots: [String]) -> [URL] {
    let fm = FileManager.default
    var out: [URL] = []
    for r in roots {
        let url = URL(fileURLWithPath: (r as NSString).expandingTildeInPath)
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else {
            FileHandle.standardError.write(Data("warning: no such path \(r)\n".utf8))
            continue
        }
        if !isDir.boolValue {
            if url.lastPathComponent == "samples.csv" {
                out.append(url.deletingLastPathComponent())
            }
            continue
        }
        if fm.fileExists(atPath: url.appendingPathComponent("samples.csv").path) {
            out.append(url)
            continue
        }
        guard let en = fm.enumerator(at: url, includingPropertiesForKeys: nil) else { continue }
        for case let f as URL in en where f.lastPathComponent == "samples.csv" {
            out.append(f.deletingLastPathComponent())
        }
    }
    return out.sorted { $0.path < $1.path }
}

func loadRun(_ dir: URL) -> Run? {
    let name = dir.lastPathComponent
    guard let text = try? String(contentsOf: dir.appendingPathComponent("samples.csv"),
                                 encoding: .utf8) else { return nil }
    var lines = text.split(separator: "\n", omittingEmptySubsequences: true)
        .map(String.init)
    guard lines.count > 1 else { return nil }
    let header = lines.removeFirst().split(separator: ",", omittingEmptySubsequences: false)
        .map { $0.trimmingCharacters(in: .whitespaces) }
    var idx: [String: Int] = [:]
    for (i, h) in header.enumerated() { idx[h] = i }

    // Schema v3 dropped gaze_cam_* from samples.csv, and the camera-frame gaze
    // cannot be recovered from raw_pitch/raw_yaw without R_n, which is not
    // logged. Such bundles simply cannot be replayed.
    guard idx["gaze_cam_x"] != nil, idx["gaze_cam_y"] != nil, idx["gaze_cam_z"] != nil else {
        FileHandle.standardError.write(Data(
            "skip \(name): no gaze_cam_* columns (samples.csv schema v3+ is lossy here)\n".utf8))
        return nil
    }
    guard let iCell = idx["cell_idx"], let iTx = idx["target_x"],
          let iTy = idx["target_y"], let iPhase = idx["phase"] else {
        FileHandle.standardError.write(Data("skip \(name): missing required columns\n".utf8))
        return nil
    }
    let gx = idx["gaze_cam_x"]!, gy = idx["gaze_cam_y"]!, gz = idx["gaze_cam_z"]!

    // Screen size comes from meta.json; without it the centre-relative
    // conversion is guesswork and the run is unusable.
    var screen = CGSize.zero
    var rows = 0, cols = 0
    if let md = try? Data(contentsOf: dir.appendingPathComponent("meta.json")),
       let obj = try? JSONSerialization.jsonObject(with: md) as? [String: Any] {
        if let s = obj["screen"] as? [String: Any],
           let w = (s["w_pt"] as? NSNumber)?.doubleValue,
           let h = (s["h_pt"] as? NSNumber)?.doubleValue {
            screen = CGSize(width: w, height: h)
        }
        if let g = obj["grid"] as? [String: Any] {
            rows = (g["rows"] as? NSNumber)?.intValue ?? 0
            cols = (g["cols"] as? NSNumber)?.intValue ?? 0
        }
    }
    guard screen.width > 0, screen.height > 0 else {
        FileHandle.standardError.write(Data("skip \(name): meta.json has no screen size\n".utf8))
        return nil
    }
    let halfW = Double(screen.width) * 0.5
    let halfH = Double(screen.height) * 0.5

    var frames: [Frame] = []
    frames.reserveCapacity(lines.count)
    for (n, line) in lines.enumerated() {
        let f = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
        guard f.count > max(gz, max(iCell, max(iTx, max(iTy, iPhase)))) else { continue }
        // Only the scored window. The dwell phase contains the saccade onto
        // the cell, which is not a fixation on it.
        guard f[iPhase].trimmingCharacters(in: .whitespaces) == "capture" else { continue }
        guard let cell = Int(f[iCell]),
              let x = Double(f[gx]), let y = Double(f[gy]), let z = Double(f[gz]),
              let tx = Double(f[iTx]), let ty = Double(f[iTy]) else { continue }
        let g = simd_double3(x, y, z)
        guard g.x.isFinite, g.y.isFinite, g.z.isFinite,
              simd_length(g) > 1e-9 else { continue }
        frames.append(Frame(cellIdx: cell,
                            gazeCam: simd_normalize(g),
                            targetRel: simd_double2(tx - halfW, ty - halfH),
                            order: n))
    }
    guard !frames.isEmpty else {
        FileHandle.standardError.write(Data("skip \(name): no capture-phase frames\n".utf8))
        return nil
    }
    return Run(name: name, screenSize: screen, frames: frames, rows: rows, cols: cols)
}

// MARK: - Fitting

/// Build a 9-dot calibration out of grid fixations.
///
/// Each frame is assigned to the standard calibration target nearest to *its
/// own cell centre* — the position we know the participant was looking at —
/// not to the target nearest its prediction. Using the prediction would let
/// the estimator's error decide its own grouping.
func fitModel(frames: [Frame], screenSize: CGSize)
    -> (model: CalibrationModel, rejected: [Int])? {
    let targets = ScreenMapper.standardTargets(screenSize: screenSize)
    var buckets = [[simd_double3]](repeating: [], count: targets.count)
    for f in frames {
        var best = 0
        var bestD = Double.infinity
        for (i, t) in targets.enumerated() {
            let d = simd_distance(t, f.targetRel)
            if d < bestD { bestD = d; best = i }
        }
        buckets[best].append(f.gazeCam)
    }
    let perDot = buckets.enumerated().map {
        (gazes: $0.element, target: targets[$0.offset])
    }
    guard let q = ScreenMapper.fitRobust(perDot: perDot) else { return nil }
    let offsets = CalibrationModel.residualOffsets(
        perDot: perDot, translation: q.translation,
        rejected: Set(q.rejectedDotIndices))
    let m = CalibrationModel(translation: q.translation,
                             targets: targets,
                             offsets: offsets,
                             fitQuality: q)
    return (m, q.rejectedDotIndices)
}

func variant(_ base: CalibrationModel,
             _ mode: CalibrationModel.LocalCorrection,
             sigma: Double?) -> CalibrationModel {
    var m = base
    m.localCorrection = mode
    m.rbfSigmaPoints = sigma
    return m
}

// MARK: - Scoring

struct CellScore {
    let run: String
    let cell: Int
    let n: Int
    let meanErrNN: Double
    let meanErrRBF: Double
}

struct Scored {
    var perFrameNN: [Double] = []
    var perFrameRBF: [Double] = []
    var cells: [CellScore] = []
    var rejectedDots = 0
    var runsUsed = 0
    var runsFailed = 0
}

func score(runs: [Run], fitFraction: Double, sigma: Double?, heldOut: Bool) -> Scored {
    var out = Scored()
    for run in runs {
        let byCell = Dictionary(grouping: run.frames, by: \.cellIdx)
        var fitFrames: [Frame] = []
        var scoreFrames: [Frame] = []
        if heldOut {
            // Temporal split within each cell: the fit never sees a frame it
            // is later scored on, while every anchor keeps training data.
            for (_, fs) in byCell {
                let sorted = fs.sorted { $0.order < $1.order }
                let cut = max(1, min(sorted.count - 1,
                                     Int((Double(sorted.count) * fitFraction).rounded())))
                fitFrames += sorted[..<cut]
                scoreFrames += sorted[cut...]
            }
        } else {
            fitFrames = run.frames
            scoreFrames = run.frames
        }
        guard scoreFrames.count > 0,
              let (base, rejected) = fitModel(frames: fitFrames,
                                              screenSize: run.screenSize) else {
            out.runsFailed += 1
            continue
        }
        out.runsUsed += 1
        out.rejectedDots += rejected.count

        let nn = variant(base, .nearestNeighbour, sigma: sigma)
        let rbf = variant(base, .gaussianRBF, sigma: sigma)

        var cellAcc: [Int: (n: Int, sNN: Double, sRBF: Double)] = [:]
        for f in scoreFrames {
            let pNN = nn.predict(gazeCam: f.gazeCam)
            let pRBF = rbf.predict(gazeCam: f.gazeCam)
            guard pNN.x.isFinite, pNN.y.isFinite,
                  pRBF.x.isFinite, pRBF.y.isFinite else { continue }
            let eNN = simd_distance(pNN, f.targetRel)
            let eRBF = simd_distance(pRBF, f.targetRel)
            out.perFrameNN.append(eNN)
            out.perFrameRBF.append(eRBF)
            var a = cellAcc[f.cellIdx] ?? (0, 0, 0)
            a.n += 1; a.sNN += eNN; a.sRBF += eRBF
            cellAcc[f.cellIdx] = a
        }
        for (cell, a) in cellAcc where a.n > 0 {
            out.cells.append(CellScore(run: run.name, cell: cell, n: a.n,
                                       meanErrNN: a.sNN / Double(a.n),
                                       meanErrRBF: a.sRBF / Double(a.n)))
        }
    }
    return out
}

// MARK: - Statistics

func mean(_ x: [Double]) -> Double {
    x.isEmpty ? .nan : x.reduce(0, +) / Double(x.count)
}
func median(_ x: [Double]) -> Double {
    guard !x.isEmpty else { return .nan }
    let s = x.sorted()
    let m = s.count / 2
    return s.count % 2 == 1 ? s[m] : (s[m - 1] + s[m]) * 0.5
}
func percentile(_ x: [Double], _ p: Double) -> Double {
    guard !x.isEmpty else { return .nan }
    let s = x.sorted()
    let i = max(0, min(s.count - 1, Int((Double(s.count - 1) * p).rounded())))
    return s[i]
}
/// Left-pad / right-pad helpers. `String(format:)` with `%s` takes a C
/// string; handing it a Swift `String` bridges to a temporary whose buffer is
/// dead by the time printf reads it, which segfaults intermittently. Do the
/// padding in Swift instead.
func pad(_ s: String, _ w: Int) -> String {
    s.count >= w ? s : s + String(repeating: " ", count: w - s.count)
}
func lpad(_ s: String, _ w: Int) -> String {
    s.count >= w ? s : String(repeating: " ", count: w - s.count) + s
}

func stddev(_ x: [Double]) -> Double {
    guard x.count > 1 else { return .nan }
    let m = mean(x)
    return (x.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(x.count - 1)).squareRoot()
}

/// Paired difference over per-cell means, with a t-statistic. Cells are the
/// unit of analysis because frames within a fixation are not independent.
func pairedReport(_ cells: [CellScore], label: String, degPerPt: Double?) {
    let d = cells.map { $0.meanErrNN - $0.meanErrRBF }   // >0 means RBF better
    guard d.count > 1 else { print("  (too few cells for a paired test)"); return }
    let md = mean(d)
    let se = stddev(d) / Double(d.count).squareRoot()
    let t = md / se
    let better = d.filter { $0 > 0 }.count
    print("  \(label): \(cells.count) cells")
    print(String(format: "    mean paired delta (NN - RBF)  %+8.3f pt   (SE %.3f, t = %+.2f)",
                 md, se, t))
    if let dpp = degPerPt {
        print(String(format: "                                  %+8.4f deg", md * dpp))
    }
    print(String(format: "    RBF better on %d / %d cells (%.0f %%)",
                 better, d.count, 100.0 * Double(better) / Double(d.count)))
    if abs(t) < 2.0 {
        print("    |t| < 2 — this is not a detectable difference at the cell level.")
    }
}

func summarise(_ s: Scored, title: String, degPerPt: Double?) {
    print("")
    print(title)
    print(String(repeating: "-", count: title.count))
    guard !s.perFrameNN.isEmpty else { print("  no scored frames"); return }
    print("  runs \(s.runsUsed) used, \(s.runsFailed) unfittable; "
          + "\(s.perFrameNN.count) frames, \(s.cells.count) cells; "
          + "\(s.rejectedDots) calibration dots rejected across all fits")
    print("  " + pad("variant", 18) + lpad("mean", 9) + lpad("median", 9)
          + lpad("p95", 9) + lpad("RMS", 9))
    for (name, e) in [("nearestNeighbour", s.perFrameNN), ("gaussianRBF", s.perFrameRBF)] {
        let rms = (e.map { $0 * $0 }.reduce(0, +) / Double(e.count)).squareRoot()
        print("  " + pad(name, 18)
              + lpad(String(format: "%.2f", mean(e)), 9)
              + lpad(String(format: "%.2f", median(e)), 9)
              + lpad(String(format: "%.2f", percentile(e, 0.95)), 9)
              + lpad(String(format: "%.2f", rms), 9))
    }
    print("  (points; lower is better)")
    pairedReport(s.cells, label: "paired over per-cell means", degPerPt: degPerPt)
}

// MARK: - Main

var paths: [String] = []
var fitFraction = 0.5
var sweep = false
var sigmaOverride: Double? = nil
var ppi: Double? = nil
var distanceMM: Double? = nil
var perCell = false
var csvOut: String? = nil

var args = Array(CommandLine.arguments.dropFirst())
var i = 0
while i < args.count {
    let a = args[i]
    switch a {
    case "--fit-fraction":
        i += 1; fitFraction = Double(args[i]) ?? 0.5
    case "--sigma":
        i += 1; sigmaOverride = Double(args[i])
    case "--sweep": sweep = true
    case "--per-cell": perCell = true
    case "--ppi": i += 1; ppi = Double(args[i])
    case "--distance-mm": i += 1; distanceMM = Double(args[i])
    case "--csv-out": i += 1; csvOut = args[i]
    case "-h", "--help":
        print("""
        usage: calibration-replay PATH [PATH ...] [options]

          PATH              a grid run bundle (dir with samples.csv + meta.json),
                            or any directory to search recursively.

          --fit-fraction F  per-cell temporal split point (default 0.5)
          --sigma S         RBF kernel width in points (default: derived from
                            target spacing, see CalibrationModel)
          --sweep           sweep sigma over a range and tabulate
          --per-cell        print the per-cell breakdown
          --ppi P           screen points per inch, for a degrees column
          --distance-mm D   eye-to-screen distance, ditto
          --csv-out FILE    write per-cell results as CSV
        """)
        exit(0)
    default:
        paths.append(a)
    }
    i += 1
}
guard !paths.isEmpty else { fail("no input paths (try --help)") }

let bundles = findBundles(paths)
guard !bundles.isEmpty else { fail("no samples.csv found under the given paths") }
let runs = bundles.compactMap(loadRun)
guard !runs.isEmpty else { fail("no replayable runs (see skip messages above)") }

var degPerPt: Double? = nil
if let p = ppi, let d = distanceMM, p > 0, d > 0 {
    degPerPt = atan((25.4 / p) / d) * 180.0 / .pi
}

print("CALIBRATION REPLAY — nearestNeighbour vs gaussianRBF")
print("bundles found   \(bundles.count)")
print("runs replayable \(runs.count)")
let totalFrames = runs.reduce(0) { $0 + $1.frames.count }
print("capture frames  \(totalFrames)")
let sampleTargets = ScreenMapper.standardTargets(screenSize: runs[0].screenSize)
let autoSigma = CalibrationModel.defaultSigmaPoints(targets: sampleTargets)
print(String(format: "default sigma   %.1f pt  (%.2f x median target spacing, screen %.0f x %.0f)",
             autoSigma, CalibrationModel.defaultSigmaSpacingMultiplier,
             runs[0].screenSize.width, runs[0].screenSize.height))
if let d = degPerPt { print(String(format: "degrees scale   1 pt = %.4f deg", d)) }
for r in runs {
    print("  \(r.name)  \(r.rows)x\(r.cols)  \(r.frames.count) frames  "
          + "\(Set(r.frames.map(\.cellIdx)).count) cells")
}

let held = score(runs: runs, fitFraction: fitFraction, sigma: sigmaOverride, heldOut: true)
summarise(held,
          title: "HELD-OUT  (per-cell temporal split at \(fitFraction)) — headline",
          degPerPt: degPerPt)

let insample = score(runs: runs, fitFraction: fitFraction, sigma: sigmaOverride, heldOut: false)
summarise(insample, title: "IN-SAMPLE  (fit and score on all frames) — reference",
          degPerPt: degPerPt)

if perCell {
    print("")
    print("PER-CELL (held-out)")
    print("  " + pad("run", 34) + lpad("cell", 5) + lpad("n", 6)
          + lpad("NN", 9) + lpad("RBF", 9) + lpad("delta", 9))
    for c in held.cells.sorted(by: { ($0.run, $0.cell) < ($1.run, $1.cell) }) {
        print("  " + pad(c.run, 34) + lpad("\(c.cell)", 5) + lpad("\(c.n)", 6)
              + lpad(String(format: "%.2f", c.meanErrNN), 9)
              + lpad(String(format: "%.2f", c.meanErrRBF), 9)
              + lpad(String(format: "%+.2f", c.meanErrNN - c.meanErrRBF), 9))
    }
}

if sweep {
    print("")
    print("SIGMA SWEEP (held-out)")
    let nnBaseline = mean(held.perFrameNN)
    print(String(format: "  nearestNeighbour baseline: %.2f pt", nnBaseline))
    print("  " + lpad("sigma_pt", 10) + lpad("x_space", 8) + lpad("mean_pt", 10)
          + lpad("delta_pt", 10) + lpad("cells_won", 12))
    let spacing = autoSigma / max(1e-9, CalibrationModel.defaultSigmaSpacingMultiplier)
    for mult in [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0] {
        let s = spacing * mult
        let r = score(runs: runs, fitFraction: fitFraction, sigma: s, heldOut: true)
        let m = mean(r.perFrameRBF)
        let won = r.cells.filter { $0.meanErrNN > $0.meanErrRBF }.count
        print("  " + lpad(String(format: "%.1f", s), 10)
              + lpad(String(format: "%.2f", mult), 8)
              + lpad(String(format: "%.2f", m), 10)
              + lpad(String(format: "%+.2f", nnBaseline - m), 10)
              + lpad("\(won)/\(r.cells.count)", 12))
    }
    print("  delta_pt > 0 means the RBF beat nearest-neighbour at that width.")
    print("  As sigma -> 0 the RBF converges to nearest-neighbour by construction,")
    print("  so the smallest rows are a sanity check, not a result.")
}

if let out = csvOut {
    var text = "run,cell,n,mean_err_nn_pt,mean_err_rbf_pt,delta_pt\n"
    for c in held.cells {
        text += "\(c.run),\(c.cell),\(c.n),"
            + String(format: "%.4f,%.4f,%.4f\n",
                     c.meanErrNN, c.meanErrRBF, c.meanErrNN - c.meanErrRBF)
    }
    try? text.write(toFile: out, atomically: true, encoding: .utf8)
    print("\nwrote \(out)")
}
