#!/usr/bin/env python3
"""
scrape_antares.py
─────────────────
Fetches all Locus and Alert properties from the ANTARES API and writes
them to properties.json.

Usage
─────
    python scrape_antares.py              # basic scrape
    python scrape_antares.py --embed      # also compute embeddings for semantic search
    python scrape_antares.py --out ../    # write to a different directory

Requirements
────────────
    pip install requests
    pip install sentence-transformers     # only needed for --embed

Semantic search notes
─────────────────────
With --embed, each property gets an 'embedding' field (384-dim float32 vector
from all-MiniLM-L6-v2). The HTML page loads these and runs a matching model
in the browser via Transformers.js — no server needed.

The embedding text is: "<name>: <description>" (falls back to just name if
description is empty). This means queries like "is the source brightening?"
will match magnitude/time properties even without exact keyword overlap.

properties.json grows from ~100 KB to ~1.4 MB with embeddings — still fast
to load, and cached by the browser after the first visit.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOCUS_URL = "https://api.antares.noirlab.edu/v1/loci_properties"
ALERT_URL = "https://api.antares.noirlab.edu/v1/alert_properties"
PAGE_SIZE = 100


# ═══════════════════════════════════════════════════════════
# FETCH
# ═══════════════════════════════════════════════════════════
def fetch_all(session, start_url):
    """Follow links.next until exhausted. Returns flat list of JSON:API items."""
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
    Flatten JSON:API items to plain dicts.
    The property name is item["id"] — not inside attributes.
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


# ═══════════════════════════════════════════════════════════
# EMBEDDINGS
# ═══════════════════════════════════════════════════════════
def embed_properties(all_props, model_name="all-MiniLM-L6-v2"):
    """
    Add a normalised 384-dim embedding to each property dict.
    Encodes "<name>: <description>" (or just name if no description).
    Embeddings are stored as lists of float32 rounded to 5 decimal places.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit(
            "\nERROR: sentence-transformers not installed.\n"
            "Run:  pip install sentence-transformers\n"
            "Then re-run with --embed."
        )

    print(f"\n[embed] Loading model '{model_name}' (downloads ~90 MB on first use, then cached)…")
    model = SentenceTransformer(model_name)

    texts = [
        f"{p['name']}: {p['description']}" if p["description"] else p["name"]
        for p in all_props
    ]

    print(f"[embed] Encoding {len(texts)} properties…")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors → dot product = cosine sim
        convert_to_numpy=True,
    )

    for prop, emb in zip(all_props, embeddings):
        prop["embedding"] = [round(float(x), 5) for x in emb]

    print(f"[embed] ✓ Embeddings computed ({embeddings.shape[1]}-dim)")
    return all_props


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Scrape ANTARES properties to properties.json")
    parser.add_argument("--out",   default=".", help="Output directory (default: current dir)")
    parser.add_argument("--embed", action="store_true",
                        help="Compute sentence embeddings for semantic search (requires sentence-transformers)")
    args = parser.parse_args()

    out_path = Path(args.out) / "properties.json"
    print("ANTARES Properties Scraper")
    print(f"Output  → {out_path.resolve()}")
    print(f"Embed   → {'yes (semantic search enabled)' if args.embed else 'no  (add --embed to enable semantic search)'}\n")

    try:
        import requests
    except ImportError:
        sys.exit("ERROR: 'requests' not installed.  Run:  pip install requests")

    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "antares-properties-scraper/2.2"})

    print(f"[1/2] Locus properties  {LOCUS_URL}")
    locus_raw = fetch_all(session, f"{LOCUS_URL}?page[limit]={PAGE_SIZE}&sort=id")
    print(f"  ✓ {len(locus_raw)} items\n")

    print(f"[2/2] Alert properties  {ALERT_URL}")
    alert_raw = fetch_all(session, f"{ALERT_URL}?page[limit]={PAGE_SIZE}&sort=id")
    print(f"  ✓ {len(alert_raw)} items\n")

    locus = normalise(locus_raw)
    alert = normalise(alert_raw)

    if args.embed:
        all_props = locus + alert
        embed_properties(all_props, model_name="all-MiniLM-L6-v2")
        # all_props was modified in-place; locus/alert share the same dicts

    payload = {
        "generated":      datetime.now(timezone.utc).isoformat(),
        "source":         "https://antares.noirlab.edu/properties",
        "has_embeddings": args.embed,
        "locus":          locus,
        "alert":          alert,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    size_kb = out_path.stat().st_size / 1024
    print(f"\n✓ Wrote {len(locus)} locus + {len(alert)} alert properties")
    print(f"  File size: {size_kb:.0f} KB")
    print(f"  → {out_path.resolve()}")

    if not args.embed:
        print("\n  Tip: run with --embed to enable natural-language semantic search")
        print("       pip install sentence-transformers && python scrape_antares.py --embed")

    print("\nNext steps:")
    print("  1. Place properties.json next to antares-properties.html")
    print("  2. git add + commit + push")
    print("  3. Local preview: python -m http.server 8000")


if __name__ == "__main__":
    main()
