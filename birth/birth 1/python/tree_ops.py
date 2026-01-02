#!/usr/bin/env python3
"""
tree_ops.py

A complete, ready-to-run script for reading and operating on a JSON tree shaped like:

{
  "exported_at": "...",
  "roots": [ { "id": ..., "name": ..., "children": [ ... ] } ]
}

Features:
- Load JSON from a file
- Print the tree
- Flatten the tree into rows (id, name, type, parent_id, path)
- Find a node by id or by name substring
- List leaf nodes
- Count nodes by type
- Validate parent_writing_id consistency (best-effort)
- Re-parent (move) a node under a new parent
- Save modified JSON (optional)

Usage examples:
  python tree_ops.py data.json print
  python tree_ops.py data.json flatten --out flat.csv
  python tree_ops.py data.json find --id 25
  python tree_ops.py data.json find --name "verb"
  python tree_ops.py data.json leaves
  python tree_ops.py data.json count-types
  python tree_ops.py data.json validate
  python tree_ops.py data.json move --node-id 25 --new-parent-id 10 --save data_modified.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


Json = Dict[str, Any]


def load_json(path: str) -> Json:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Json, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def walk(node: Json, visit_fn: Callable[[Json, Optional[Json], List[str]], None],
         parent: Optional[Json] = None, path_parts: Optional[List[str]] = None) -> None:
    """
    Depth-first walk. Calls visit_fn(node, parent, path_parts_including_node_name).
    """
    if path_parts is None:
        path_parts = []
    current_path = path_parts + [str(node.get("name", ""))]
    visit_fn(node, parent, current_path)
    for child in node.get("children", []) or []:
        walk(child, visit_fn, node, current_path)


def print_tree(node: Json, depth: int = 0) -> None:
    name = node.get("name", "<unnamed>")
    node_id = node.get("id", "?")
    node_type = node.get("type", None)
    type_str = f" ({node_type})" if node_type is not None else ""
    print("  " * depth + f"- {name} [id={node_id}]{type_str}")
    for child in node.get("children", []) or []:
        print_tree(child, depth + 1)


@dataclass
class FlatRow:
    id: Any
    name: str
    type: Optional[str]
    parent_id: Any
    path: str


def flatten(roots: List[Json], sep: str = " > ") -> List[FlatRow]:
    rows: List[FlatRow] = []

    def collect(n: Json, p: Optional[Json], path_parts: List[str]) -> None:
        rows.append(
            FlatRow(
                id=n.get("id"),
                name=str(n.get("name", "")),
                type=n.get("type"),
                parent_id=(p.get("id") if p else None),
                path=sep.join([pp for pp in path_parts if pp]),
            )
        )

    for r in roots:
        walk(r, collect)

    return rows


def write_csv(rows: List[FlatRow], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "type", "parent_id", "path"])
        for r in rows:
            w.writerow([r.id, r.name, r.type, r.parent_id, r.path])


def find_by_id(roots: List[Json], target_id: Any) -> Optional[Json]:
    found: Optional[Json] = None

    def finder(n: Json, p: Optional[Json], path_parts: List[str]) -> None:
        nonlocal found
        if found is not None:
            return
        if n.get("id") == target_id:
            found = n

    for r in roots:
        walk(r, finder)

    return found


def find_with_parent_and_path(roots: List[Json], predicate: Callable[[Json], bool]) -> List[Tuple[Json, Optional[Json], str]]:
    matches: List[Tuple[Json, Optional[Json], str]] = []

    def collect(n: Json, p: Optional[Json], path_parts: List[str]) -> None:
        if predicate(n):
            matches.append((n, p, " > ".join([pp for pp in path_parts if pp])))

    for r in roots:
        walk(r, collect)

    return matches


def leaves(roots: List[Json]) -> List[Tuple[Json, str]]:
    out: List[Tuple[Json, str]] = []

    def collect(n: Json, p: Optional[Json], path_parts: List[str]) -> None:
        children = n.get("children", []) or []
        if len(children) == 0:
            out.append((n, " > ".join([pp for pp in path_parts if pp])))

    for r in roots:
        walk(r, collect)

    return out


def count_types(roots: List[Json]) -> Counter:
    c: Counter = Counter()

    def collect(n: Json, p: Optional[Json], path_parts: List[str]) -> None:
        if "type" in n and n["type"] is not None:
            c[n["type"]] += 1
        else:
            c["<missing>"] += 1

    for r in roots:
        walk(r, collect)

    return c


def validate_parent_writing_id(roots: List[Json]) -> List[str]:
    """
    Best-effort validation:
    - For each node with parent_writing_id field, check it matches actual parent's id.
    - Also ensure ids are unique.
    """
    errors: List[str] = []
    seen_ids = set()

    def check(n: Json, p: Optional[Json], path_parts: List[str]) -> None:
        node_id = n.get("id")
        path = " > ".join([pp for pp in path_parts if pp])

        if node_id in seen_ids:
            errors.append(f"Duplicate id={node_id} at path: {path}")
        else:
            seen_ids.add(node_id)

        if "parent_writing_id" in n:
            expected = p.get("id") if p else None
            actual = n.get("parent_writing_id")
            if actual != expected:
                errors.append(
                    f"parent_writing_id mismatch at id={node_id} ({path}): "
                    f"actual={actual!r}, expected={expected!r}"
                )

    for r in roots:
        walk(r, check)

    return errors


def detach_node(roots: List[Json], node_id: Any) -> Tuple[Optional[Json], Optional[Json]]:
    """
    Removes node from its current parent's children (or roots if it's a root).
    Returns (node, old_parent). If node is a root, old_parent is None.
    """
    # Check roots first
    for i, r in enumerate(list(roots)):
        if r.get("id") == node_id:
            node = roots.pop(i)
            return node, None

    # Search parents
    def find_parent_and_index(n: Json, p: Optional[Json], path_parts: List[str]) -> None:
        nonlocal found_parent, found_index
        if found_parent is not None:
            return
        children = n.get("children", []) or []
        for idx, ch in enumerate(children):
            if ch.get("id") == node_id:
                found_parent = n
                found_index = idx
                return

    found_parent: Optional[Json] = None
    found_index: Optional[int] = None

    for r in roots:
        walk(r, find_parent_and_index)

    if found_parent is None or found_index is None:
        return None, None

    node = (found_parent.get("children") or []).pop(found_index)
    return node, found_parent


def attach_node(roots: List[Json], node: Json, new_parent_id: Optional[Any]) -> None:
    """
    Attaches node under new_parent_id. If new_parent_id is None, attaches as a root.
    Updates node['parent_writing_id'] if that field exists or if you want it created.
    """
    if new_parent_id is None:
        roots.append(node)
        node["parent_writing_id"] = None
        return

    parent = find_by_id(roots, new_parent_id)
    if parent is None:
        raise ValueError(f"New parent id={new_parent_id} not found.")

    parent.setdefault("children", [])
    parent["children"].append(node)
    node["parent_writing_id"] = new_parent_id


def is_descendant(node: Json, potential_descendant_id: Any) -> bool:
    """
    True if potential_descendant_id occurs anywhere under node.
    Prevents cycles when moving nodes.
    """
    found = False

    def finder(n: Json, p: Optional[Json], path_parts: List[str]) -> None:
        nonlocal found
        if found:
            return
        if n.get("id") == potential_descendant_id:
            found = True

    walk(node, finder)
    return found


def move_node(roots: List[Json], node_id: Any, new_parent_id: Optional[Any]) -> None:
    node = find_by_id(roots, node_id)
    if node is None:
        raise ValueError(f"Node id={node_id} not found.")

    if new_parent_id is not None:
        new_parent = find_by_id(roots, new_parent_id)
        if new_parent is None:
            raise ValueError(f"New parent id={new_parent_id} not found.")
        if is_descendant(node, new_parent_id):
            raise ValueError("Invalid move: would create a cycle (new parent is inside node's subtree).")

    detached, _old_parent = detach_node(roots, node_id)
    if detached is None:
        raise ValueError(f"Failed to detach node id={node_id} (not found).")

    attach_node(roots, detached, new_parent_id)


def cmd_print(args: argparse.Namespace, data: Json) -> int:
    for r in data.get("roots", []):
        print_tree(r)
    return 0


def cmd_flatten(args: argparse.Namespace, data: Json) -> int:
    rows = flatten(data.get("roots", []))
    if args.out:
        write_csv(rows, args.out)
        print(f"Wrote {len(rows)} rows to {args.out}")
    else:
        # Print as TSV to stdout
        print("id\tname\ttype\tparent_id\tpath")
        for r in rows:
            print(f"{r.id}\t{r.name}\t{r.type}\t{r.parent_id}\t{r.path}")
    return 0


def cmd_find(args: argparse.Namespace, data: Json) -> int:
    roots = data.get("roots", [])
    if args.id is not None:
        matches = find_with_parent_and_path(roots, lambda n: n.get("id") == args.id)
    else:
        needle = (args.name or "").lower()
        matches = find_with_parent_and_path(
            roots,
            lambda n: needle in str(n.get("name", "")).lower()
        )

    if not matches:
        print("No matches.")
        return 1

    for n, p, path in matches:
        print(f"- id={n.get('id')} name={n.get('name')!r} type={n.get('type')!r} parent_id={(p.get('id') if p else None)!r}")
        print(f"  path: {path}")
    return 0


def cmd_leaves(args: argparse.Namespace, data: Json) -> int:
    ls = leaves(data.get("roots", []))
    for n, path in ls:
        print(f"- id={n.get('id')} name={n.get('name')!r} path: {path}")
    print(f"Total leaves: {len(ls)}")
    return 0


def cmd_count_types(args: argparse.Namespace, data: Json) -> int:
    c = count_types(data.get("roots", []))
    for k, v in c.most_common():
        print(f"{k}: {v}")
    return 0


def cmd_validate(args: argparse.Namespace, data: Json) -> int:
    errs = validate_parent_writing_id(data.get("roots", []))
    if not errs:
        print("OK: no validation errors found.")
        return 0
    print("Validation errors:")
    for e in errs:
        print(f"- {e}")
    return 2


def cmd_move(args: argparse.Namespace, data: Json) -> int:
    roots = data.get("roots", [])
    move_node(roots, args.node_id, args.new_parent_id)

    if args.save:
        save_json(data, args.save)
        print(f"Moved node id={args.node_id} under parent id={args.new_parent_id} and saved to {args.save}")
    else:
        print(f"Moved node id={args.node_id} under parent id={args.new_parent_id}. (Not saved; use --save)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Operate on a JSON tree export.")
    p.add_argument("json_file", help="Path to JSON file (e.g., data.json)")

    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("print", help="Print the tree")
    sp.set_defaults(func=cmd_print)

    sp = sub.add_parser("flatten", help="Flatten the tree to rows (optionally CSV)")
    sp.add_argument("--out", help="Write CSV to this path (otherwise prints TSV to stdout)")
    sp.set_defaults(func=cmd_flatten)

    sp = sub.add_parser("find", help="Find nodes by id or name substring")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int, help="Find node with this id")
    g.add_argument("--name", help="Case-insensitive substring match on name")
    sp.set_defaults(func=cmd_find)

    sp = sub.add_parser("leaves", help="List leaf nodes (nodes with no children)")
    sp.set_defaults(func=cmd_leaves)

    sp = sub.add_parser("count-types", help="Count nodes by 'type' value (or <missing>)")
    sp.set_defaults(func=cmd_count_types)

    sp = sub.add_parser("validate", help="Validate ids are unique and parent_writing_id matches structure")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("move", help="Move a node under a new parent (or to root if new parent is omitted)")
    sp.add_argument("--node-id", type=int, required=True, help="ID of node to move")
    sp.add_argument("--new-parent-id", type=int, default=None, help="ID of new parent (omit to move to root)")
    sp.add_argument("--save", help="Write modified JSON to this path")
    sp.set_defaults(func=cmd_move)

    return p


def main(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data = load_json(args.json_file)
    if "roots" not in data or not isinstance(data["roots"], list):
        print("Error: JSON must have a top-level 'roots' list.", file=sys.stderr)
        return 2

    return args.func(args, data)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
