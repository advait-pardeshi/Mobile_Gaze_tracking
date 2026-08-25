import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = GazeViewModel()
    @State private var showGridPicker = false
    @State private var showModelFetch = false

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .top) {
                CameraPreview(
                    session: viewModel.cameraManager.session,
                    landmarks: viewModel.landmarks,
                    imageSize: viewModel.imageSize,
                    headPose: viewModel.headPose,
                    intrinsics: viewModel.cameraManager.intrinsics,
                    gazeOriginCam: viewModel.gazeOriginCam,
                    gazeDirCam: viewModel.gaze?.gazeCam
                )
                .ignoresSafeArea()

                VStack(spacing: 4) {
                    HStack {
                        Text("Faces: \(viewModel.faceCount)")
                        Spacer()
                        Text("Landmarks: \(viewModel.landmarks.count)")
                        Spacer()
                        Text(String(format: "FPS: %.0f", viewModel.fps))
                    }
                    if let pose = viewModel.headPose {
                        let e = pose.euler
                        HStack {
                            Text(String(format: "Yaw: %+5.1f°", e.yaw))
                            Spacer()
                            Text(String(format: "Pitch: %+5.1f°", e.pitch))
                            Spacer()
                            Text(String(format: "Roll: %+5.1f°", e.roll))
                            Spacer()
                            Text(String(format: "Z: %4.0fmm", pose.translation.z))
                        }
                    } else {
                        HStack {
                            Text("Head pose: — (\(viewModel.headPoseFailureReason ?? "?"))")
                            Spacer()
                        }
                        if let s = viewModel.intrinsicsSummary {
                            HStack { Text(s); Spacer() }
                        }
                    }
                    HStack {
                        Text(viewModel.gazeEstimatorLoaded ? "Model: ✓ loaded" : "Model: ✗ not found")
                        Spacer()
                    }
                    if let g = viewModel.gaze {
                        let toDeg = 180.0 / .pi
                        HStack {
                            Text(String(format: "Gaze pitch: %+5.1f°", g.pitch * toDeg))
                            Spacer()
                            Text(String(format: "Gaze yaw: %+5.1f°", g.yaw * toDeg))
                        }
                    } else {
                        HStack {
                            Text("Gaze: — (waiting for head pose)")
                            Spacer()
                        }
                    }
                    HStack {
                        Text(calibrationStatusText)
                        Spacer()
                    }
                }
                .font(.system(.footnote, design: .monospaced))
                .foregroundColor(.green)
                .padding(8)
                .background(Color.black.opacity(0.55))
                .padding(.top, 4)

                // Stage 3 viz: hidden during any full-screen run so it doesn't
                // compete with the stimulus for attention.
                if let img = viewModel.normalizedEyes, viewModel.isIdle {
                    VStack(spacing: 2) {
                        Text("Stage 3 — normalized eyes (left | right)")
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundColor(.green)
                        Image(uiImage: img)
                            .interpolation(.none)
                            .resizable()
                            .frame(width: 400, height: 100)
                            .border(Color.green, width: 1)
                    }
                    .padding(8)
                    .background(Color.black.opacity(0.55))
                    .frame(maxHeight: .infinity, alignment: .bottom)
                    .padding(.bottom, 8)
                }

                // Stage 5: live gaze dot, only when calibration is fitted
                // and no calibration/accuracy/grid run is in progress, and
                // no results screen is up. Experiment 2 deliberately keeps
                // the dot visible (drawn inside its own overlay) so the
                // user can see the model prediction during traversal.
                if let p = viewModel.gazeScreenPoint, viewModel.isIdle {
                    Circle()
                        .strokeBorder(Color.white.opacity(0.85), lineWidth: 2)
                        .background(Circle().fill(Color.cyan.opacity(0.85)))
                        .frame(width: 26, height: 26)
                        .shadow(color: .black.opacity(0.6), radius: 3)
                        .position(p)
                        .allowsHitTesting(false)
                }

                // Stage 5 calibration + validation + experiment triggers
                // (stacked).
                if viewModel.isIdle {
                    VStack(alignment: .trailing, spacing: 8) {
                        if viewModel.isCalibrationUsable {
                            triggerButton("Experiment 4 — Predictive (cued)",
                                          color: .indigo.opacity(0.9)) {
                                viewModel.startPredictiveTask(scoring: .cued)
                            }
                            triggerButton("Experiment 4 — Predictive (free)",
                                          color: .indigo.opacity(0.6)) {
                                viewModel.startPredictiveTask(scoring: .free)
                            }
                            triggerButton("Experiment 3 — Communication",
                                          color: .pink.opacity(0.9)) {
                                viewModel.startCommunicationTask()
                            }
                            triggerButton("Experiment 2 — Fixation",
                                          color: .orange.opacity(0.9)) {
                                viewModel.startFixationExperiment()
                            }
                            triggerButton("Experiment 1 — Grid",
                                          color: .purple.opacity(0.85)) {
                                showGridPicker = true
                            }
                            triggerButton("Collect Fine-tune Data",
                                          color: .yellow.opacity(0.9),
                                          textColor: .black) {
                                viewModel.startFineTuneCollection()
                            }
                            triggerButton("Accuracy Test",
                                          color: .red.opacity(0.85)) {
                                viewModel.startAccuracyTest()
                            }
                        }
                        // Validation and (re-)calibration stay reachable even
                        // when the gate above is closed — otherwise a poor
                        // verdict would be an unrecoverable dead end.
                        if viewModel.calibration != nil {
                            triggerButton("Validate Calibration",
                                          color: .teal.opacity(0.9)) {
                                viewModel.startCalibrationValidation()
                            }
                        }
                        if let reason = viewModel.calibrationBlockReason {
                            Text(reason)
                                .font(.caption2)
                                .foregroundColor(.yellow)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.black.opacity(0.55))
                                .cornerRadius(6)
                        }
                        triggerButton(viewModel.calibration == nil
                                      ? "Calibrate" : "Re-calibrate",
                                      color: .black.opacity(0.7)) {
                            viewModel.startCalibration()
                        }
                        triggerButton("Fetch Model",
                                      color: .gray.opacity(0.85)) {
                            showModelFetch = true
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity,
                           alignment: .bottomTrailing)
                    .padding(.trailing, 16)
                    .padding(.bottom, viewModel.normalizedEyes == nil ? 24 : 130)
                    .confirmationDialog(
                        "Experiment 1 — grid focus",
                        isPresented: $showGridPicker,
                        titleVisibility: .visible
                    ) {
                        ForEach(GridSize.presets) { size in
                            Button(buttonLabel(for: size)) {
                                viewModel.startGridExperiment(size)
                            }
                        }
                        Button("Cancel", role: .cancel) {}
                    }
                }

                // Stage 5 calibration overlay.
                if let c = viewModel.calibrationController {
                    CalibrationOverlay(
                        controller: c,
                        screenSize: geo.size,
                        onCancel: { viewModel.cancelCalibration() }
                    )
                }

                // Accuracy test overlay (active run).
                if let a = viewModel.accuracyController {
                    AccuracyTestOverlay(
                        controller: a,
                        screenSize: geo.size,
                        onCancel: { viewModel.cancelAccuracyTest() }
                    )
                }

                // Accuracy test results screen (after a run completes).
                if let r = viewModel.accuracyResult {
                    AccuracyResultsView(
                        result: r,
                        screenSize: geo.size,
                        onDismiss: { viewModel.dismissAccuracyResult() },
                        onRerun: {
                            viewModel.dismissAccuracyResult()
                            viewModel.startAccuracyTest()
                        }
                    )
                }

                // Experiment 1: active run overlay.
                if let ge = viewModel.gridExperimentController {
                    GridExperimentOverlay(
                        controller: ge,
                        screenSize: geo.size,
                        onCancel: { viewModel.cancelGridExperiment() }
                    )
                }

                // Experiment 1: results screen.
                if let r = viewModel.gridExperimentResult {
                    GridExperimentResultsView(
                        result: r,
                        screenSize: geo.size,
                        onDismiss: { viewModel.dismissGridExperimentResult() },
                        onRerun: {
                            let size = r.gridSize
                            viewModel.dismissGridExperimentResult()
                            viewModel.startGridExperiment(size)
                        }
                    )
                }

                // Calibration validation: active run (all 9 dots at once).
                if let v = viewModel.validationController {
                    CalibrationValidationOverlay(
                        controller: v,
                        screenSize: geo.size,
                        livePrediction: viewModel.gazeScreenPoint,
                        onCancel: { viewModel.cancelCalibrationValidation() },
                        onFinish: { v.finishNow() }
                    )
                }

                // Calibration validation: verdict screen. This is the gate
                // before any experiment runs.
                if let r = viewModel.validationResult {
                    CalibrationValidationResultsView(
                        result: r,
                        screenSize: geo.size,
                        onAccept: { viewModel.acceptValidation() },
                        onRecalibrate: {
                            viewModel.discardCalibrationAndRecalibrate()
                        },
                        onRerun: {
                            viewModel.acceptValidation()
                            viewModel.startCalibrationValidation()
                        }
                    )
                }

                // Experiment 2: fixation-stability overlay.
                if let f = viewModel.fixationController {
                    FixationStabilityOverlay(
                        controller: f,
                        screenSize: geo.size,
                        livePrediction: viewModel.gazeScreenPoint,
                        onCancel: { viewModel.cancelFixationExperiment() }
                    )
                }

                // Experiment 2: results screen.
                if let r = viewModel.fixationResult {
                    FixationStabilityResultsView(
                        result: r,
                        screenSize: geo.size,
                        onDismiss: { viewModel.dismissFixationResult() },
                        onRerun: {
                            viewModel.dismissFixationResult()
                            viewModel.startFixationExperiment()
                        }
                    )
                }

                // Experiment 3: communication-task overlay (word grid).
                if let ct = viewModel.commTaskController {
                    CommunicationTaskOverlay(
                        controller: ct,
                        screenSize: geo.size,
                        livePrediction: viewModel.gazeScreenPoint,
                        onCancel: { viewModel.cancelCommunicationTask() },
                        onFinish: { ct.finishNow() }
                    )
                }

                // Experiment 4: prompted predictive overlay (3×3 grid).
                if let pt = viewModel.predictiveController {
                    PredictiveTaskOverlay(
                        controller: pt,
                        screenSize: geo.size,
                        livePrediction: viewModel.gazeScreenPoint,
                        onCancel: { viewModel.cancelPredictiveTask() },
                        onFinish: { pt.finishNow() }
                    )
                }

                // Experiments 3 and 4 have no results screen by design: the run ends
                // straight back to idle and the numbers are read off the
                // master log / run bundle instead of the phone.
            }
            .onAppear {
                viewModel.screenSize = geo.size
                viewModel.start()
            }
            .onChange(of: geo.size) { newSize in
                viewModel.screenSize = newSize
            }
            .onDisappear { viewModel.stop() }
            .sheet(isPresented: $showModelFetch) {
                ModelFetchSheet(
                    viewModel: viewModel,
                    onDismiss: { showModelFetch = false }
                )
            }
        }
    }

    /// HUD line for calibration state. Surfaces the validation verdict, not
    /// just "ready" — a fitted-but-unvalidated calibration is exactly the
    /// situation the validation step exists to make visible.
    private var calibrationStatusText: String {
        guard viewModel.calibration != nil else { return "Calibration: —" }
        guard let v = viewModel.lastValidationVerdict else {
            return "Calibration: fitted (not validated)"
        }
        let deg = viewModel.lastValidationMeanDegrees
        let degText = deg.isFinite ? String(format: " %.2f°", deg) : ""
        return "Calibration: \(v.rawValue)\(degText)"
    }

    /// One trigger button in the right-hand stack. Factored out because the
    /// stack has eight of them and they differ only in title, colour and
    /// action.
    private func triggerButton(_ title: String,
                               color: Color,
                               textColor: Color = .white,
                               action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.callout.weight(.semibold))
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(color)
                .foregroundColor(textColor)
                .cornerRadius(10)
        }
    }

    /// Format one row of the grid-size picker, e.g. "4×4  (16 trials · ~1.0 min)".
    private func buttonLabel(for size: GridSize) -> String {
        String(format: "%@  (%d trials · ~%.1f min)",
               size.label, size.trialCount, size.estimatedMinutes)
    }
}
