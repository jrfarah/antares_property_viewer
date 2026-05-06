#!/usr/bin/env python3
"""
scrape_antares.py
─────────────────
Fetches all Locus and Alert properties from the ANTARES API and writes
them to properties.json, which antares-properties.html loads as static data.

Usage
─────
    python scrape_antares.py              # writes properties.json here
    python scrape_antares.py --out ../    # write to a different directory

Requirements
────────────
    pip install requests

JSON:API response shape (confirmed):
    {
      "data": [
        {
          "id": "num_alerts",              ← this is the property name
          "type": "locus_property",
          "attributes": {
            "origin": "ANTARES",
            "description": "Number of total Alerts on a Locus.",
            "type": "int"
          }
        }
      ],
      "links": { "next": "...url..." },
      "meta":  { "count": 267 }
    }
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOCUS_URL = "https://api.antares.noirlab.edu/v1/loci_properties?page[limit]=100&sort=id"
ALERT_URL = "https://api.antares.noirlab.edu/v1/alert_properties?page[limit]=100&sort=id"


def fetch_all(session, start_url):
    """Follow links.next until exhausted. Returns flat list of raw JSON:API items."""
    results = []
    url = start_url
    page = 0

    while url:
        page += 1
        print(f"    GET {url}  (page {page})", flush=True)
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("data", [])
        results.extend(batch)
        print(f"    → {len(batch)} items  (total: {len(results)} / {data.get('meta',{}).get('count','?')})")

        url = data.get("links", {}).get("next")
        if url:
            time.sleep(0.1)

    return results


def normalise(items):
    """
    Map JSON:API items to plain { name, type, origin, description } dicts.
    The property name lives in item["id"], not in attributes.
    """
    out = []
    for item in items:
        attrs = item.get("attributes", {})
        out.append({
            "name":        str(item.get("id")          or ""),
            "type":        str(attrs.get("type")        or "").lower(),
            "origin":      str(attrs.get("origin")      or "ANTARES"),
            "description": str(attrs.get("description") or ""),
        })
    out.sort(key=lambda x: x["name"].lower())
    return out


def main():
    parser = argparse.ArgumentParser(description="Scrape ANTARES properties to properties.json")
    parser.add_argument("--out", default=".", help="Output directory (default: current dir)")
    args = parser.parse_args()

    out_path = Path(args.out) / "properties.json"
    print("ANTARES Properties Scraper")
    print(f"Output → {out_path.resolve()}\n")

    try:
        import requests
    except ImportError:
        sys.exit("ERROR: 'requests' not installed.  Run:  pip install requests")

    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "antares-properties-scraper/2.1"})

    print(f"[1/2] Locus properties")
    locus_raw = fetch_all(session, LOCUS_URL)
    print(f"  ✓ {len(locus_raw)} items\n")

    print(f"[2/2] Alert properties")
    alert_raw = fetch_all(session, ALERT_URL)
    print(f"  ✓ {len(alert_raw)} items\n")

    locus = normalise(locus_raw)
    alert = normalise(alert_raw)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source":    "https://antares.noirlab.edu/properties",
        "locus":     locus,
        "alert":     alert,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"✓ Wrote {len(locus)} locus + {len(alert)} alert properties")
    print(f"  → {out_path.resolve()}")
    print("\nNext steps:")
    print("  1. Place properties.json next to antares-properties.html")
    print("  2. git add + commit + push")
    print("  3. Local preview: python -m http.server 8000")


if __name__ == "__main__":
    main()
