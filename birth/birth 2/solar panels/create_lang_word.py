#!/usr/bin/env python3
"""
Recursively find all `word.json` files and compile them into one `lang_word.json`
with the structure:

{
  "lang_word": "solar panels",
  "children": [ ...all loaded JSON objects... ]
}

Notes:
- If a word.json contains a single object, it is appended as one child.
- If a word.json contains a list of objects, each object is appended.
- Optional flags for strict mode, dedupe, and provenance.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


# -----------------------------
# HARD-CODED TOP-LEVEL VALUE
# -----------------------------
LANG_WORD_VALUE = "solar panels"


JsonObj = Dict[str, Any]
JsonRoot = Union[JsonObj, List[Any]]


def _load_json(path: Path) -> JsonRoot:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_children_items(data: JsonRoot, path: Path) -> List[JsonObj]:
    """
    Normalize word.json content into a list of objects to append into `children`.
    Accepts either:
      - a single object
      - a list of objects
    """
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        # ensure list items are objects (dicts)
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: list item at index {i} is not an object")
        return data
    raise ValueError(f"{path}: JSON root must be an object or a list of objects")


@dataclass
class Options:
    root: Path
    out: Path
    strict: bool
    include_source_path: bool
    dedupe: bool


def _dedupe_key(obj: JsonObj) -> Tuple[str, str]:
    """
    Dedupe key using parent.word + parent.type when available.
    Falls back to empty strings if not present.
    """
    parent = obj.get("parent", {}) if isinstance(obj.get("parent", {}), dict) else {}
    w = str(parent.get("word", "")).strip()
    t = str(parent.get("type", "")).strip()
    return (w, t)


def compile_to_lang_word(opts: Options) -> JsonObj:
    word_files = sorted(opts.root.rglob("word.json"))

    children: List[JsonObj] = []
    seen = set()
    errors: List[Dict[str, Any]] = []

    for p in word_files:
        try:
            raw = _load_json(p)
            items = _as_children_items(raw, p)

            for obj in items:
                if opts.include_source_path:
                    obj = dict(obj)  # shallow copy
                    obj["_source"] = os.fspath(p.relative_to(opts.root))

                if opts.dedupe:
                    k = _dedupe_key(obj)
                    if k in seen:
                        continue
                    seen.add(k)

                children.append(obj)

        except Exception as e:
            errors.append({"file": os.fspath(p), "error": str(e)})
            if opts.strict:
                raise

    # Your requested final structure:
    result: JsonObj = {
        "lang_word": LANG_WORD_VALUE,
        "children": children,
    }

    # Optional: keep errors in a non-intrusive field (comment out if you don't want it)
    if errors:
        result["_errors"] = errors

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compile all word.json files into a single lang_word.json"
    )
    ap.add_argument("root", help="Root directory to scan recursively")
    ap.add_argument(
        "-o",
        "--out",
        default="lang_word.json",
        help="Output file path (default: ./lang_word.json)",
    )
    ap.add_argument(
        "--include-source-path",
        action="store_true",
        help="Add '_source' to each child indicating where it came from",
    )
    ap.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate entries based on (parent.word, parent.type) when present",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately if any word.json is invalid",
    )
    args = ap.parse_args()

    opts = Options(
        root=Path(args.root).expanduser().resolve(),
        out=Path(args.out).expanduser().resolve(),
        strict=bool(args.strict),
        include_source_path=bool(args.include_source_path),
        dedupe=bool(args.dedupe),
    )

    if not opts.root.is_dir():
        raise SystemExit(f"Invalid root directory: {opts.root}")

    result = compile_to_lang_word(opts)

    opts.out.parent.mkdir(parents=True, exist_ok=True)
    with opts.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {opts.out}")
    print(f"word.json files found: {len(list(opts.root.rglob('word.json')))}")
    print(f"children compiled: {len(result['children'])}")
    if "_errors" in result:
        print(f"errors: {len(result['_errors'])}")


if __name__ == "__main__":
    main()
