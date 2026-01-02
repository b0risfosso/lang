#!/usr/bin/env python3
"""
Link nodes in a tree JSON to matching rows in the City of Chicago Contracts dataset (rsxa-ify5)
using keyword searches across:
- purchase_order_description
- department
- vendor_name
- contract_type
- specification_number

Usage:
  python link_nodes_to_chicago_contracts.py \
      --in tree.json \
      --out tree_linked.json \
      --sample 5 \
      --min-term-len 3 \
      --max-terms 8

Optional:
  - Set SOCRATA_APP_TOKEN env var for better rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

DATASET_ID = "rsxa-ify5"
SOCRATA_BASE = "https://data.cityofchicago.org/resource"
SOCRATA_ENDPOINT = f"{SOCRATA_BASE}/{DATASET_ID}.json"

SEARCH_FIELDS = [
    "purchase_order_description",
    "department",
    "vendor_name",
    "contract_type",
    "specification_number",
]

# ---- term generation helpers ----

STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with",
    "by", "from", "at", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "as", "into", "over", "under",
    "city",  # NOTE: we handle "city of chicago" explicitly as a special case
}

# A small synonym/expansion map. Add your own as you discover patterns.
SYNONYMS: Dict[str, List[str]] = {
    "solar": ["photovoltaic", "pv"],
    "microgrid": ["micro grid", "distributed energy", "distributed generation"],
    "panels": ["panel"],
    "contract": ["agreement"],
    "chicago": ["city of chicago"],
}

SPECIAL_CASE_TERMS: Dict[str, List[str]] = {
    # Example from your prompt:
    "solar panels": ["solar", "photovoltaic", "pv", "panel"],
    # Your other example:
    "city of chicago": ["city of chicago"],  # special handling (see below)
}


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(name: str) -> List[str]:
    # keep letters/numbers; split on everything else
    tokens = re.split(r"[^a-z0-9]+", name.lower())
    tokens = [t for t in tokens if t and t not in STOPWORDS]
    return tokens


def expand_terms_from_name(name: str, min_term_len: int, max_terms: int) -> List[str]:
    n = normalize_text(name)

    # Special cases first (exact normalized match)
    if n in SPECIAL_CASE_TERMS:
        return SPECIAL_CASE_TERMS[n][:max_terms]

    tokens = tokenize(n)

    # Expand with synonyms
    terms: List[str] = []
    for t in tokens:
        if len(t) < min_term_len:
            continue
        terms.append(t)
        for syn in SYNONYMS.get(t, []):
            terms.append(syn)

    # Also include the full phrase if it's reasonably short and informative
    if len(n) >= min_term_len and len(n.split()) <= 6:
        terms.append(n)

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for t in terms:
        t2 = normalize_text(t)
        if len(t2) < min_term_len:
            continue
        if t2 in seen:
            continue
        seen.add(t2)
        out.append(t2)
        if len(out) >= max_terms:
            break

    return out


# ---- Socrata query building ----

def escape_like(term: str) -> str:
    # escape % and _ for LIKE; Socrata uses SQL-ish syntax
    term = term.replace("\\", "\\\\")
    term = term.replace("%", "\\%").replace("_", "\\_")
    term = term.replace("'", "''")
    return term


def build_where_clause(terms: List[str]) -> str:
    """
    Build a $where clause matching any term across any SEARCH_FIELDS.
    Uses: lower(field) like '%term%'
    """
    if not terms:
        return "1=0"  # no terms -> match nothing

    or_clauses: List[str] = []
    for term in terms:
        et = escape_like(term)
        field_clauses = [
            f"lower({field}) like '%{et}%' escape '\\\\'"
            for field in SEARCH_FIELDS
        ]
        or_clauses.append("(" + " OR ".join(field_clauses) + ")")

    return "(" + " OR ".join(or_clauses) + ")"


@dataclass
class ContractMatch:
    count: int
    sample_rows: List[Dict[str, Any]]
    where: str
    terms: List[str]


def socrata_get(params: Dict[str, str], app_token: Optional[str]) -> List[Dict[str, Any]]:
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    r = requests.get(SOCRATA_ENDPOINT, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def get_match_count(where: str, app_token: Optional[str]) -> int:
    rows = socrata_get(
        {
            "$select": "count(*) as c",
            "$where": where,
        },
        app_token,
    )
    if not rows:
        return 0
    try:
        return int(rows[0].get("c", 0))
    except (TypeError, ValueError):
        return 0


def get_sample_rows(where: str, sample: int, app_token: Optional[str]) -> List[Dict[str, Any]]:
    # Pull only a small set of fields for inspection (add more if you like)
    select_fields = [
        "purchase_order_contract_number",
        "revision_number",
        "specification_number",
        "purchase_order_description",
        "department",
        "vendor_name",
        "contract_type",
        "approval_date",
        "start_date",
        "end_date",
    ]
    rows = socrata_get(
        {
            "$select": ", ".join(select_fields),
            "$where": where,
            "$limit": str(sample),
        },
        app_token,
    )
    return rows


# ---- Tree traversal & enrichment ----

def iter_nodes(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a flat list of all nodes under each root."""
    out: List[Dict[str, Any]] = []

    def walk(node: Dict[str, Any]) -> None:
        out.append(node)
        for child in node.get("children", []) or []:
            walk(child)

    for r in root.get("roots", []) or []:
        walk(r)

    return out


def enrich_tree(
    data: Dict[str, Any],
    sample: int,
    min_term_len: int,
    max_terms: int,
    app_token: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    nodes = iter_nodes(data)

    summary = {
        "dataset_id": DATASET_ID,
        "endpoint": SOCRATA_ENDPOINT,
        "fields_searched": SEARCH_FIELDS,
        "nodes_total": len(nodes),
        "nodes_linked": 0,
    }

    for node in nodes:
        name = str(node.get("name", "")).strip()
        if not name:
            continue

        n_norm = normalize_text(name)

        # Special handling for "city of chicago" example:
        # If node is exactly "city of chicago", link to ALL rows by using where=1=1
        if n_norm == "city of chicago":
            where = "1=1"
            terms = ["<ALL_ROWS>"]
        else:
            terms = expand_terms_from_name(name, min_term_len=min_term_len, max_terms=max_terms)
            where = build_where_clause(terms)

        try:
            count = get_match_count(where, app_token=app_token)
            sample_rows = get_sample_rows(where, sample=sample, app_token=app_token) if count else []
        except requests.HTTPError as e:
            # Store the error on the node and keep going
            node["chicago_contracts_link"] = {
                "dataset_id": DATASET_ID,
                "error": f"HTTPError: {e}",
                "terms": terms,
                "where": where,
                "match_count": 0,
                "sample_rows": [],
            }
            continue
        except requests.RequestException as e:
            node["chicago_contracts_link"] = {
                "dataset_id": DATASET_ID,
                "error": f"RequestException: {e}",
                "terms": terms,
                "where": where,
                "match_count": 0,
                "sample_rows": [],
            }
            continue

        node["chicago_contracts_link"] = {
            "dataset_id": DATASET_ID,
            "terms": terms,
            "where": where,
            "match_count": count,
            "sample_rows": sample_rows,
        }
        summary["nodes_linked"] += 1

    return data, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input tree JSON file")
    ap.add_argument("--out", dest="out", required=True, help="Output enriched JSON file")
    ap.add_argument("--sample", type=int, default=5, help="Sample rows to attach per node")
    ap.add_argument("--min-term-len", type=int, default=3, help="Minimum term length")
    ap.add_argument("--max-terms", type=int, default=8, help="Max terms per node")
    args = ap.parse_args()

    app_token = os.environ.get("SOCRATA_APP_TOKEN")

    with open(args.inp, "r", encoding="utf-8") as f:
        data = json.load(f)

    enriched, summary = enrich_tree(
        data,
        sample=args.sample,
        min_term_len=args.min_term_len,
        max_terms=args.max_terms,
        app_token=app_token,
    )

    # Attach top-level provenance about the linkage operation
    enriched.setdefault("chicago_contracts_linkage", {})
    enriched["chicago_contracts_linkage"].update({
        "dataset_id": DATASET_ID,
        "endpoint": SOCRATA_ENDPOINT,
        "fields_searched": SEARCH_FIELDS,
        "run_with_app_token": bool(app_token),
        "summary": summary,
    })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"Wrote enriched JSON -> {args.out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
