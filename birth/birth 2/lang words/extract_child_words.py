import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

INPUT_PATH = "/Users/b/fantasiagenesis/lang/birth/birth 2/lang words/lang_word_solar_panels.json"  # change if needed


def walk_collect_words(node: Any, path: Tuple[str, ...], out: List[Tuple[str, str]]) -> None:
    """
    Recursively walk a JSON-like structure and collect every object that has a 'word' field.
    Each collected word is tagged with a 'type_path' derived from the JSON keys we traversed.
    """
    if isinstance(node, dict):
        # If this dict itself has a word, collect it under the current path
        if "word" in node and isinstance(node["word"], str):
            type_path = ".".join(path) if path else "(root)"
            out.append((type_path, node["word"]))

        # Traverse children/other keys
        for k, v in node.items():
            # Don't treat the literal key name "word" as part of the path
            if k == "word":
                continue
            walk_collect_words(v, path + (k,), out)

    elif isinstance(node, list):
        for item in node:
            walk_collect_words(item, path, out)


def compile_seed_word_summary(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []

    for entry in data.get("children", []):
        seed = entry.get("seed_word") or {}
        seed_word = seed.get("word", "")
        seed_type = seed.get("type", "")

        collected: List[Tuple[str, str]] = []
        walk_collect_words(entry.get("children", {}), tuple(), collected)

        # Build a flat list in the exact "type_path: child_word" form
        flat = [f"{type_path}: {word}" for type_path, word in collected]

        # Optional: also group by type_path
        grouped = defaultdict(list)
        for type_path, word in collected:
            grouped[type_path].append(word)

        results.append(
            {
                "seed_word": seed_word,
                "seed_word_type": seed_type,
                "flat_children": flat,              # e.g. ["popular_searches: LNG", ...]
                "grouped_children": dict(grouped),  # e.g. {"popular_searches": ["LNG", ...], ...}
            }
        )

    return results


def main() -> None:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = compile_seed_word_summary(data)

    # Print a readable version
    for item in summary:
        print(f'\n=== {item["seed_word"]}: {item["seed_word_type"]} ===')
        for s in item["flat_children"]:
            print(f"- {s}")

    # If you want JSON output instead, uncomment:
    # print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
