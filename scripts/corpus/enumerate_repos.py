#!/usr/bin/env python3
"""Enumerate public GitHub repositories containing q source, with licences.

qqWen's pretraining corpus was 1.6M tokens from 15 repos. The open question for
building anything better is whether the licence-filtered public corpus is 30x
that or 3x. This answers it by enumeration rather than guesswork.

GitHub's search API returns at most 1000 results per query, and there are ~1426
q repos, so the search is sliced by star count and the slices are deduped.
Unauthenticated search allows ~10 requests/minute, so this is slow but polite.

    python scripts/corpus/enumerate_repos.py --out out/repos.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com/search/repositories"
# Slices chosen to keep every bucket under the 1000-result cap.
STAR_SLICES = [">=100", "20..99", "5..19", "2..4", "1", "0"]
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc",
              "unlicense", "cc0-1.0", "0bsd", "mit-0"}


def get(url: str, retries: int = 6) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "q-corpus-survey",
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):          # secondary rate limit
                wait = 60 * (attempt + 1)
                print(f"  rate limited, sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as exc:
            print(f"  network error {exc}, retrying", flush=True)
            time.sleep(10)
    raise RuntimeError(f"giving up on {url}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/repos.json")
    args = ap.parse_args()

    repos: dict[str, dict] = {}
    for stars in STAR_SLICES:
        page = 1
        while True:
            url = (f"{API}?q=language:q+stars:{stars}&per_page=100&page={page}"
                   f"&sort=updated")
            data = get(url)
            items = data.get("items", [])
            for it in items:
                lic = (it.get("license") or {}).get("key")
                repos[it["full_name"]] = {
                    "full_name": it["full_name"],
                    "license": lic,
                    "permissive": lic in PERMISSIVE,
                    "size_kb": it.get("size", 0),
                    "stars": it.get("stargazers_count", 0),
                    "fork": it.get("fork", False),
                    "pushed_at": it.get("pushed_at"),
                    "clone_url": it["clone_url"],
                }
            total = data.get("total_count", 0)
            print(f"stars {stars} page {page}: +{len(items)} "
                  f"(running total {len(repos)} of ~{total} in slice)", flush=True)
            if len(items) < 100 or page * 100 >= min(total, 1000):
                break
            page += 1
            time.sleep(6)          # ~10 requests/minute unauthenticated
        time.sleep(6)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(repos.values(), key=lambda r: -r["stars"]), indent=1))

    perm = [r for r in repos.values() if r["permissive"]]
    unlicensed = [r for r in repos.values() if not r["license"]]
    print(f"\n{len(repos)} repos; {len(perm)} permissively licensed, "
          f"{len(unlicensed)} with no licence (unusable), "
          f"{len(repos) - len(perm) - len(unlicensed)} other licences")
    print(f"permissive size on disk (all files, not just .q): "
          f"{sum(r['size_kb'] for r in perm) / 1e6:.2f} GB")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
