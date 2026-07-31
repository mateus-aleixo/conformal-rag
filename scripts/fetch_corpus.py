"""Fetch the public-domain corpus (US Army technical manuals, 17 U.S.C. §105).

Sources are Internet Archive items. PDFs land in data/raw/ and are never
committed. Run:  python scripts/fetch_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

RAW = Path(__file__).parent.parent / "data" / "raw"

# (filename, archive.org download URL) — US federal government works, public
# domain under 17 U.S.C. §105. Item IDs verified against the metadata API.
MANUALS = [
    (
        "TM-9-8000-principles-of-automotive-vehicles-1985.pdf",
        "https://archive.org/download/tm-9-8000-principles-of-automotive-vehicles-1985/TM9-8000_Principles_of_automotive_vehicles_1985.pdf",
    ),
    (
        "TM-9-6115-641-24P-5kW-MEP802-generator-set.pdf",
        "https://archive.org/download/5kw-mep-802-tqg-9-6115-641-24-p-nov-12-pin-071613/5kw%20MEP802%20TQG%209-6115-641-24P%20%20Nov%2012%20pin%20071613.pdf",
    ),
]


def fetch(name: str, url: str) -> bool:
    dest = RAW / name
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"already present: {name}")
        return True
    print(f"fetching {name} ...")
    try:
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=120,
            headers={"User-Agent": "conformal-rag corpus fetcher"},
        ) as resp:
            resp.raise_for_status()
            RAW.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes(1 << 16):
                    f.write(chunk)
    except httpx.HTTPError as e:
        print(f"  FAILED: {e}", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return False
    print(f"  ok: {dest.stat().st_size / 1e6:.1f} MB")
    return True


if __name__ == "__main__":
    results = [fetch(name, url) for name, url in MANUALS]
    sys.exit(0 if all(results) else 1)
