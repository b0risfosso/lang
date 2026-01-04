import json

# Load JSON file
with open("/Users/b/fantasiagenesis/lang/birth/birth 2/lang words/lang_word_solar_panels.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract parent word + type
seeds = [
    {
        "word": item["seed_word"]["word"],
        "type": item["seed_word"]["type"]
    }
    for item in data.get("children", [])
    if "seed_word" in item
]

# Output result
print(seeds)
