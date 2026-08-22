#!/usr/bin/env python3
"""
Serve a fine-tuned `.mlpackage` to the iOS app over LAN HTTP so the phone
can hot-swap weights without rebuilding.

Usage:
    python3 serve_model.py path/to/GazeNet.mlpackage [--port 8000]

The script:
  1. Walks the .mlpackage directory and writes a `manifest.json` inside it.
     The iOS `ModelUpdater` GETs that file first to learn what else to
     download.
  2. cd's to the package's parent directory and starts http.server.
  3. Prints the exact URL the phone needs (paste it into the app's
     "Fetch Model" sheet).

Press Ctrl-C to stop.
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler


# Our manifest's filename is deliberately not `manifest.json`: macOS APFS
# is case-insensitive by default, and a real `.mlpackage` already contains
# Apple's `Manifest.json` (capital M) at its root. Writing `manifest.json`
# would collide and corrupt the package's CoreML manifest.
SERVE_MANIFEST_NAME = "_serve_manifest.json"


def build_manifest(pkg: pathlib.Path) -> dict:
    files: list[dict] = []
    for root, _, names in os.walk(pkg):
        for n in names:
            if n == SERVE_MANIFEST_NAME:
                continue  # don't list ourselves
            full = pathlib.Path(root) / n
            rel = full.relative_to(pkg)
            files.append({
                "path": str(rel).replace(os.sep, "/"),
                "size": full.stat().st_size,
            })
    files.sort(key=lambda e: e["path"])
    return {"package_name": pkg.name, "files": files}


def lan_ip() -> str:
    """Best-effort local IP discovery: open a UDP socket and read the
    OS-chosen source address. Doesn't actually send any packets."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mlpackage", type=pathlib.Path,
                    help="Path to GazeNet.mlpackage (a directory).")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    pkg = args.mlpackage.resolve()
    if not pkg.is_dir():
        print(f"[serve] error: {pkg} is not a directory (.mlpackage expected).")
        return 1

    manifest = build_manifest(pkg)
    manifest_path = pkg / SERVE_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[serve] manifest:  {manifest_path}  ({len(manifest['files'])} files)")

    os.chdir(pkg.parent)
    ip = lan_ip()
    base = f"http://{ip}:{args.port}/{pkg.name}/"
    print(f"[serve] Phone URL: {base}")
    print(f"[serve] Press Ctrl-C to stop.")

    httpd = HTTPServer(("0.0.0.0", args.port), SimpleHTTPRequestHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
