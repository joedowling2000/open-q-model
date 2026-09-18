#!/usr/bin/env python3
"""Shallow-clone permissively licensed q repos and extract their q source.

Keeps only MIT/Apache/BSD/ISC/CC0-style licences, so whatever is trained on this
can be released without licence questions. Repos with no licence at all are
excluded: "public on GitHub" is not a grant of rights.

Each kept file is written with its provenance (repo, path, licence, commit) so
the corpus can be audited or a repo removed later. Clones are deleted as we go;
only the .q text is retained.

    python scripts/corpus/harvest.py --repos out/repos.json --out corpus/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

Q_SUFFIXES = {".q", ".k"}          # .k included: the layer q sits on
SKIP_DIRS = {".git", "node_modules", "vendor", "third_party", "build"}
MAX_FILE_BYTES = 2_000_000


def clone(url: str, dest: Path) -> str | None:
    """Shallow-clone and return the HEAD sha, or None on failure."""
    r = subprocess.run(["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return None
    sha = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    return sha.stdout.strip() or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="out/repos.json")
    ap.add_argument("--out", default="corpus/raw")
    ap.add_argument("--manifest", default="corpus/manifest.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    repos = [r for r in json.loads(Path(args.repos).read_text())
             if r["permissive"] and not r["fork"]]
    if args.limit:
        repos = repos[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if manifest.exists():                      # resumable
        done = {json.loads(l)["repo"] for l in manifest.open()}
        print(f"resuming: {len(done)} repos already harvested")

    seen_hashes: set[str] = set()
    if manifest.exists():
        seen_hashes = {json.loads(l)["sha256"] for l in manifest.open()}

    kept = bytes_kept = 0
    with manifest.open("a") as mf:
        for i, repo in enumerate(repos, 1):
            if repo["full_name"] in done:
                continue
            with tempfile.TemporaryDirectory(dir="/home/joedowling/Projects/qeval/corpus") as tmp:
                dest = Path(tmp) / "repo"
                sha = clone(repo["clone_url"], dest)
                if sha is None:
                    print(f"[{i}/{len(repos)}] {repo['full_name']}: clone failed", flush=True)
                    continue
                n_files = 0
                for path in dest.rglob("*"):
                    if not path.is_file() or path.suffix.lower() not in Q_SUFFIXES:
                        continue
                    if any(part in SKIP_DIRS for part in path.parts):
                        continue
                    try:
                        data = path.read_bytes()
                    except OSError:
                        continue
                    if not data or len(data) > MAX_FILE_BYTES:
                        continue
                    digest = hashlib.sha256(data).hexdigest()
                    if digest in seen_hashes:      # exact-duplicate across repos
                        continue
                    seen_hashes.add(digest)
                    target = out_dir / digest[:2] / f"{digest}.q"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    mf.write(json.dumps({
                        "repo": repo["full_name"], "license": repo["license"],
                        "commit": sha, "path": str(path.relative_to(dest)),
                        "sha256": digest, "bytes": len(data),
                    }) + "\n")
                    n_files += 1
                    bytes_kept += len(data)
                mf.flush()
                kept += n_files
                if n_files or i % 25 == 0:
                    print(f"[{i}/{len(repos)}] {repo['full_name']}: {n_files} q files "
                          f"(total {kept} files, {bytes_kept/1e6:.1f} MB)", flush=True)
            shutil.rmtree(Path(tmp), ignore_errors=True)

    print(f"\nharvested {kept} unique q files, {bytes_kept/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
