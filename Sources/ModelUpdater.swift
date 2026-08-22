import Foundation

/// Pulls a freshly fine-tuned `.mlpackage` from a laptop on the same Wi-Fi
/// network so we can hot-swap weights without rebuilding the app.
///
/// ## Protocol
///
/// `baseURL` points at a directory served by the laptop (e.g. by
/// `Tools/serve_model.py`). The directory **is** the `.mlpackage` bundle.
///
/// 1. GET `baseURL/manifest.json`. Expected shape:
///    ```
///    {
///      "package_name": "GazeNet.mlpackage",
///      "files": [
///        { "path": "Manifest.json", "size": 234 },
///        { "path": "Data/com.apple.CoreML/model.mlmodel", "size": 5512 },
///        { "path": "Data/com.apple.CoreML/weights/weight.bin", "size": 12345678 }
///      ]
///    }
///    ```
/// 2. For every entry, GET `baseURL/<path>` and write it to
///    `Documents/incoming_models/<stamp>.mlpackage/<path>`.
/// 3. Return the assembled `.mlpackage` URL — caller passes it to
///    `GazeEstimator.replace(modelAt:)`.
///
/// The manifest's own filename is `manifest.json`; we don't ship it back
/// into the assembled package directory.
enum ModelUpdater {

    /// Distinct filename for our serve-side manifest so it can't collide
    /// with Apple's `Manifest.json` (capital M) at the root of a real
    /// `.mlpackage`. macOS APFS is case-insensitive by default, so any
    /// lowercase `manifest.json` written inside the bundle would overwrite
    /// CoreML's manifest and corrupt the package.
    static let serveManifestName = "_serve_manifest.json"

    struct Manifest: Decodable {
        let package_name: String
        let files: [Entry]
        struct Entry: Decodable {
            let path: String
            let size: Int
        }
    }

    enum FetchError: LocalizedError {
        case badURL
        case manifestFailed(String)
        case fileFailed(path: String, reason: String)
        case writeFailed(String)

        var errorDescription: String? {
            switch self {
            case .badURL:                          return "Bad URL."
            case .manifestFailed(let s):           return "Manifest: \(s)"
            case .fileFailed(let p, let r):        return "File '\(p)': \(r)"
            case .writeFailed(let s):              return "Disk: \(s)"
            }
        }
    }

    /// Streams progress out of the fetch loop so the UI can render a bar.
    struct Progress {
        let fileIndex: Int
        let fileCount: Int
        let bytesSoFar: Int
        let bytesTotal: Int
    }

    /// Download every file in the remote `.mlpackage` and return the local
    /// assembled bundle URL. Throws on any network/disk failure; on success
    /// the destination is fully populated.
    ///
    /// `baseURL` MUST end with a trailing slash and point at the
    /// `.mlpackage` directory served over HTTP.
    static func fetch(baseURL: URL,
                      progress: ((Progress) -> Void)? = nil) async throws -> URL {
        guard baseURL.scheme == "http" || baseURL.scheme == "https" else {
            throw FetchError.badURL
        }
        let session = URLSession(configuration: .ephemeral)

        // 1. Manifest.
        let manifestURL = baseURL.appendingPathComponent(serveManifestName)
        let manifest: Manifest
        do {
            let (data, response) = try await session.data(from: manifestURL)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                throw FetchError.manifestFailed("HTTP \(http.statusCode)")
            }
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch let e as FetchError {
            throw e
        } catch {
            throw FetchError.manifestFailed(error.localizedDescription)
        }

        // 2. Destination directory.
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd_HHmmss"
        let stamp = df.string(from: Date())
        let pkgName = manifest.package_name.isEmpty
            ? "GazeNet.mlpackage" : manifest.package_name
        let docs = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
        let parent = docs.appendingPathComponent("incoming_models",
                                                  isDirectory: true)
        let dest = parent.appendingPathComponent("\(stamp)_\(pkgName)",
                                                  isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: dest,
                                                    withIntermediateDirectories: true)
        } catch {
            throw FetchError.writeFailed(error.localizedDescription)
        }

        // 3. Files.
        let totalBytes = manifest.files.reduce(0) { $0 + $1.size }
        var soFar = 0
        for (i, entry) in manifest.files.enumerated() {
            // Skip the manifest itself if the laptop accidentally included
            // it in the file list.
            if entry.path == serveManifestName { continue }
            // Reject paths that try to escape the destination dir.
            let normalized = (entry.path as NSString).standardizingPath
            if normalized.hasPrefix("/") || normalized.contains("..") {
                throw FetchError.fileFailed(path: entry.path,
                                            reason: "unsafe path")
            }
            let fileURL = baseURL.appendingPathComponent(entry.path)
            let outURL = dest.appendingPathComponent(entry.path)
            do {
                try FileManager.default.createDirectory(
                    at: outURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
            } catch {
                throw FetchError.writeFailed(error.localizedDescription)
            }
            do {
                let (data, response) = try await session.data(from: fileURL)
                if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                    throw FetchError.fileFailed(path: entry.path,
                                                reason: "HTTP \(http.statusCode)")
                }
                try data.write(to: outURL, options: .atomic)
                soFar += data.count
                progress?(Progress(fileIndex: i + 1,
                                   fileCount: manifest.files.count,
                                   bytesSoFar: soFar,
                                   bytesTotal: totalBytes))
            } catch let e as FetchError {
                throw e
            } catch {
                throw FetchError.fileFailed(path: entry.path,
                                            reason: error.localizedDescription)
            }
        }

        return dest
    }
}
