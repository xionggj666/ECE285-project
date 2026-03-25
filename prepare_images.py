import os
import re
import json
import shutil
from pathlib import Path

import kagglehub

print("Downloading dataset from KaggleHub...")
dataset_path = kagglehub.dataset_download("lsind18/tarot-json")
SRC_ROOT = Path(dataset_path)
print("Dataset downloaded to:", SRC_ROOT)

OUT_DIR = Path("data/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def find_json_file(root: Path):
    candidates = list(root.rglob("tarot-images.json"))
    if not candidates:
        raise FileNotFoundError("Cannot find tarot-images.json in dataset")
    return candidates[0]


def find_image_by_stem(root: Path, stem: str):
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    for ext in exts:
        matches = list(root.rglob(stem + ext))
        if matches:
            return matches[0]
    return None


def normalize_cards(data):
    """
    Convert loaded JSON into a unified list of dicts:
    each item should at least contain a card name.
    """
    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"name": str(item)})
        return out

    if isinstance(data, dict):
        out = []

        # case 1: {"The Fool": {...}, "The Magician": {...}}
        # case 2: {"cards": [...]}
        if "cards" in data and isinstance(data["cards"], list):
            for item in data["cards"]:
                if isinstance(item, dict):
                    out.append(item)
                else:
                    out.append({"name": str(item)})
            return out

        for k, v in data.items():
            if isinstance(v, dict):
                item = dict(v)
                if "name" not in item:
                    item["name"] = k
                out.append(item)
            else:
                out.append({"name": k, "image": v})
        return out

    raise TypeError(f"Unsupported JSON top-level type: {type(data)}")


def pick_image_candidate(card: dict, src_root: Path):
    # Try common fields first
    for key in ["img", "image", "image_url", "file", "filename", "path", "src"]:
        if key in card and card[key]:
            val = str(card[key])

            # local relative path
            potential = src_root / val
            if potential.exists():
                return potential

            # maybe basename only
            basename = Path(val).name
            for p in src_root.rglob(basename):
                if p.exists():
                    return p

    return None


def main():
    json_path = find_json_file(SRC_ROOT)
    print("Using JSON:", json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print("Top-level JSON type:", type(raw).__name__)
    if isinstance(raw, dict):
        print("Top-level keys sample:", list(raw.keys())[:10])

    cards = normalize_cards(raw)
    print("Normalized card count:", len(cards))
    if cards:
        print("First normalized sample:", json.dumps(cards[0], ensure_ascii=False, indent=2)[:500])

    copied = 0
    missing = []

    for card in cards:
        card_name = card.get("name")
        if not card_name:
            continue

        slug = safe_name(card_name)

        image_candidate = pick_image_candidate(card, SRC_ROOT)

        if image_candidate is None:
            image_candidate = find_image_by_stem(SRC_ROOT, slug)

        if image_candidate is None or not image_candidate.exists():
            missing.append(card_name)
            continue

        dst = OUT_DIR / f"{slug}.png"
        shutil.copy(image_candidate, dst)
        copied += 1
        print(f"Copied: {card_name} -> {dst.name}")

    print("\nDone.")
    print("Copied:", copied)
    print("Missing:", len(missing))

    if missing:
        print("\nMissing cards:")
        for x in missing:
            print(" -", x)


if __name__ == "__main__":
    main()