import Foundation

/// Zip a directory in place using `NSFileCoordinator(.forUploading)`.
///
/// That coordination option hands back a temporary zip of the coordinated
/// item; the archive only survives the callback, so it has to be copied out
/// before the closure returns. Shared by `FineTuneDataCollector` and
/// `ExperimentRunLog`, which both ship a run directory through the share
/// sheet.
enum DirectoryZip {

    /// Zip `directory` to `destination`, replacing any existing file there.
    /// Returns false (and logs) if coordination or the copy failed.
    @discardableResult
    static func zip(directory: URL, to destination: URL, tag: String) -> Bool {
        try? FileManager.default.removeItem(at: destination)

        let coordinator = NSFileCoordinator()
        var nsErr: NSError?
        var success = false
        coordinator.coordinate(readingItemAt: directory,
                               options: [.forUploading],
                               error: &nsErr) { tmpZip in
            do {
                try FileManager.default.copyItem(at: tmpZip, to: destination)
                success = true
            } catch {
                print("[\(tag)] zip copy failed: \(error)")
            }
        }
        if let e = nsErr {
            print("[\(tag)] coordinator error: \(e)")
        }
        return success
    }
}
