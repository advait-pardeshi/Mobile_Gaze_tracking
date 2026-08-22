import SwiftUI

/// Modal sheet for pulling a freshly fine-tuned `.mlpackage` from a laptop
/// on the local network. The URL persists between sessions in
/// UserDefaults so the user only has to type it once per laptop.
struct ModelFetchSheet: View {
    @ObservedObject var viewModel: GazeViewModel
    let onDismiss: () -> Void

    @AppStorage("model_fetch_base_url")
    private var urlString: String = "http://192.168.1.42:8000/GazeNet.mlpackage/"

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Laptop URL")) {
                    TextField("http://192.168.1.42:8000/GazeNet.mlpackage/",
                              text: $urlString)
                        .keyboardType(.URL)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                    Text("Run `python3 Tools/serve_model.py path/to/GazeNet.mlpackage` "
                         + "on the laptop. The script prints this URL.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section(header: Text("Status")) {
                    if viewModel.modelFetchInFlight {
                        HStack {
                            ProgressView()
                            Text(viewModel.modelFetchStatus ?? "Working…")
                                .font(.callout)
                        }
                    } else if let s = viewModel.modelFetchStatus {
                        Text(s).font(.callout)
                    } else {
                        Text("Current model: \(viewModel.modelSource)")
                            .font(.callout)
                            .foregroundColor(.secondary)
                    }
                }

                Section {
                    Button {
                        viewModel.fetchModel(baseURLString: urlString)
                    } label: {
                        HStack {
                            Spacer()
                            Text(viewModel.modelFetchInFlight
                                 ? "Fetching…" : "Fetch & Swap Model")
                                .font(.body.weight(.semibold))
                            Spacer()
                        }
                    }
                    .disabled(viewModel.modelFetchInFlight)
                }
            }
            .navigationTitle("Hot-swap Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done", action: onDismiss)
                }
            }
            .alert("Fine-tuned model loaded",
                   isPresented: Binding(
                       get: { viewModel.modelFetchSuccess != nil },
                       set: { if !$0 { viewModel.modelFetchSuccess = nil } })) {
                Button("OK", role: .cancel) { viewModel.modelFetchSuccess = nil }
            } message: {
                Text("\(viewModel.modelFetchSuccess ?? "") is now active. Gaze estimation is using the new weights.")
            }
        }
    }
}
