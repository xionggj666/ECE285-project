import os
import re
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import kagglehub
from PIL import Image, ImageDraw


# ============================================================
# Config
# ============================================================
DATASET_ID = "lsind18/tarot-json"
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
LAYOUTS_DIR = DATA_DIR / "layouts"
SPECS_PATH = DATA_DIR / "tarot_specs.json"
TRAIN_JSONL_PATH = DATA_DIR / "train.jsonl"
COPIED_JSON_PATH = DATA_DIR / "tarot-images.json"

W, H = 512, 768
FRAME_MARGIN = 18


# ============================================================
# Tarot card lists
# ============================================================
MAJOR_ARCANA = [
    "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
    "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
    "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
    "The Devil", "The Tower", "The Star", "The Moon", "The Sun", "Judgement", "The World",
]
SUITS = ["Wands", "Cups", "Swords", "Pentacles"]
RANKS = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Page", "Knight", "Queen", "King"]
MINOR_ARCANA = [f"{rank} of {suit}" for suit in SUITS for rank in RANKS]
TAROT_CARDS = MAJOR_ARCANA + MINOR_ARCANA


# ============================================================
# Utilities
# ============================================================
def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def rank_to_num(rank: str) -> Optional[int]:
    mapping = {
        "Ace": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
        "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9, "Ten": 10,
    }
    return mapping.get(rank)


def find_file(root: Path, filename: str) -> Optional[Path]:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def find_image_by_stem(root: Path, stem: str) -> Optional[Path]:
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    for ext in exts:
        matches = list(root.rglob(stem + ext))
        if matches:
            return matches[0]
    return None


# ============================================================
# Step 1. Download dataset and copy tarot-images.json + images
# ============================================================
def normalize_cards(data: Any) -> List[Dict[str, Any]]:
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

    raise TypeError(f"Unsupported JSON type: {type(data)}")


def pick_image_candidate(card: Dict[str, Any], src_root: Path) -> Optional[Path]:
    for key in ["img", "image", "image_url", "file", "filename", "path", "src"]:
        if key in card and card[key]:
            val = str(card[key])
            potential = src_root / val
            if potential.exists():
                return potential
            basename = Path(val).name
            for p in src_root.rglob(basename):
                if p.exists():
                    return p
    return None


def prepare_kaggle_tarot_images() -> Tuple[Path, List[Dict[str, Any]]]:
    ensure_dirs()

    print(f"Downloading dataset from KaggleHub: {DATASET_ID}")
    dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
    print("Dataset downloaded to:", dataset_path)

    json_path = find_file(dataset_path, "tarot-images.json")
    if json_path is None:
        raise FileNotFoundError("Cannot find tarot-images.json in downloaded dataset")

    shutil.copy(json_path, COPIED_JSON_PATH)
    print(f"Copied JSON: {json_path} -> {COPIED_JSON_PATH}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cards = normalize_cards(raw)

    copied = 0
    missing = []
    for card in cards:
        card_name = card.get("name")
        if not card_name:
            continue
        slug = safe_name(card_name)

        image_candidate = pick_image_candidate(card, dataset_path)
        if image_candidate is None:
            image_candidate = find_image_by_stem(dataset_path, slug)

        if image_candidate is None or not image_candidate.exists():
            missing.append(card_name)
            continue

        dst = IMAGES_DIR / f"{slug}.png"
        shutil.copy(image_candidate, dst)
        copied += 1

    print(f"Prepared images: copied={copied}, missing={len(missing)}")
    if missing:
        print("Missing images sample:", missing[:10])

    return dataset_path, cards


# ============================================================
# Step 2. Build card specs
# ============================================================
MAJOR_SPECS: Dict[str, Dict[str, Any]] = {
    "The Fool": {
        "layout_type": "single_figure_center",
        "main_figure": "traveler",
        "symbols": ["dog", "sun", "cliff", "bag", "flower"],
        "mood": "innocent adventurous",
    },
    "The Magician": {
        "layout_type": "single_figure_center",
        "main_figure": "magician",
        "symbols": ["wand", "cup", "sword", "pentacle", "table", "infinity"],
        "mood": "focused powerful",
    },
    "The High Priestess": {
        "layout_type": "single_figure_center",
        "main_figure": "priestess",
        "symbols": ["moon", "scroll", "pillars", "veil"],
        "mood": "mysterious contemplative",
    },
    "The Empress": {
        "layout_type": "single_figure_center",
        "main_figure": "empress",
        "symbols": ["crown", "wheat", "forest", "throne"],
        "mood": "abundant nurturing",
    },
    "The Emperor": {
        "layout_type": "single_figure_center",
        "main_figure": "emperor",
        "symbols": ["throne", "scepter", "mountains", "ram"],
        "mood": "stable authoritative",
    },
    "The Hierophant": {
        "layout_type": "single_figure_center",
        "main_figure": "teacher",
        "symbols": ["staff", "keys", "acolytes"],
        "mood": "solemn traditional",
    },
    "The Lovers": {
        "layout_type": "double_figure",
        "main_figure": "lovers",
        "symbols": ["angel", "tree", "sun", "serpent"],
        "mood": "harmonious fateful",
    },
    "The Chariot": {
        "layout_type": "vehicle_center",
        "main_figure": "charioteer",
        "symbols": ["chariot", "two beasts", "canopy"],
        "mood": "driven controlled",
    },
    "Strength": {
        "layout_type": "figure_animal",
        "main_figure": "tamer",
        "symbols": ["lion", "infinity", "flowers"],
        "mood": "calm courage",
    },
    "The Hermit": {
        "layout_type": "single_figure_center",
        "main_figure": "hermit",
        "symbols": ["lantern", "staff", "mountain"],
        "mood": "wise introspective",
    },
    "Wheel of Fortune": {
        "layout_type": "symbol_center",
        "main_figure": "wheel",
        "symbols": ["wheel", "sphinx", "serpent", "clouds"],
        "mood": "cyclical cosmic",
    },
    "Justice": {
        "layout_type": "single_figure_center",
        "main_figure": "judge",
        "symbols": ["scales", "sword", "throne"],
        "mood": "balanced exact",
    },
    "The Hanged Man": {
        "layout_type": "suspended_figure",
        "main_figure": "hanged man",
        "symbols": ["tree", "halo"],
        "mood": "surrendered enlightened",
    },
    "Death": {
        "layout_type": "figure_mount",
        "main_figure": "skeletal rider",
        "symbols": ["horse", "flag", "sunrise"],
        "mood": "grave transformative",
    },
    "Temperance": {
        "layout_type": "single_figure_center",
        "main_figure": "angel",
        "symbols": ["two cups", "water", "path"],
        "mood": "balanced healing",
    },
    "The Devil": {
        "layout_type": "triangle_top",
        "main_figure": "devil",
        "symbols": ["chains", "two captives", "torch"],
        "mood": "oppressive tempting",
    },
    "The Tower": {
        "layout_type": "tower_scene",
        "main_figure": "tower",
        "symbols": ["lightning", "fire", "falling figures"],
        "mood": "catastrophic sudden",
    },
    "The Star": {
        "layout_type": "single_figure_bottom",
        "main_figure": "maiden",
        "symbols": ["large star", "small stars", "water"],
        "mood": "hopeful serene",
    },
    "The Moon": {
        "layout_type": "path_scene",
        "main_figure": "moon path",
        "symbols": ["moon", "dog", "wolf", "crayfish", "towers"],
        "mood": "dreamlike uncertain",
    },
    "The Sun": {
        "layout_type": "single_figure_center",
        "main_figure": "radiant child",
        "symbols": ["sun", "sunflowers", "horse"],
        "mood": "bright triumphant",
    },
    "Judgement": {
        "layout_type": "top_bottom_scene",
        "main_figure": "angel",
        "symbols": ["trumpet", "coffins", "people"],
        "mood": "awakening redemptive",
    },
    "The World": {
        "layout_type": "symbol_center",
        "main_figure": "dancing figure",
        "symbols": ["wreath", "four creatures", "batons"],
        "mood": "complete harmonious",
    },
}


def build_minor_spec(card_name: str) -> Dict[str, Any]:
    m = re.match(r"^(Ace|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Page|Knight|Queen|King) of (Wands|Cups|Swords|Pentacles)$", card_name)
    if not m:
        raise ValueError(f"Invalid minor arcana name: {card_name}")

    rank, suit = m.group(1), m.group(2)
    n = rank_to_num(rank)
    suit_symbol = suit[:-1].lower()

    if rank in ["Page", "Knight", "Queen", "King"]:
        layout_type = "court_figure"
        main_figure = rank.lower()
        symbols = [suit_symbol, "throne" if rank in ["Queen", "King"] else "banner"]
        mood = {
            "Page": "curious youthful",
            "Knight": "dynamic determined",
            "Queen": "calm composed",
            "King": "authoritative stable",
        }[rank]
    elif rank == "Ace":
        layout_type = "symbol_center"
        main_figure = f"single {suit_symbol}"
        symbols = [suit_symbol, "halo", "cloud"]
        mood = "pure potential"
    else:
        layout_type = "multi_symbol"
        main_figure = "none"
        symbols = [suit_symbol] * (n if n is not None else 2)
        mood = "symbolic composition"

    return {
        "card_name": card_name,
        "arcana": "minor",
        "suit": suit.lower(),
        "rank": rank.lower(),
        "layout_type": layout_type,
        "main_figure": main_figure,
        "symbols": symbols,
        "count": n,
        "mood": mood,
        "slug": safe_name(card_name),
    }


def build_all_specs(cards_from_json: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    if cards_from_json:
        card_names = []
        for item in cards_from_json:
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                card_names.append(name.strip())
        if not card_names:
            card_names = TAROT_CARDS
    else:
        card_names = TAROT_CARDS

    specs = []
    for card in card_names:
        if card in MAJOR_SPECS:
            spec = {
                "card_name": card,
                "arcana": "major",
                "suit": None,
                "rank": None,
                **MAJOR_SPECS[card],
                "slug": safe_name(card),
            }
        else:
            spec = build_minor_spec(card)
        specs.append(spec)

    with open(SPECS_PATH, "w", encoding="utf-8") as f:
        json.dump(specs, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(specs)} specs -> {SPECS_PATH}")
    return specs


# ============================================================
# Step 3. Render layout control images
# ============================================================
COLORS = {
    "background": (0, 0, 0),
    "frame": (255, 255, 255),
    "figure": (220, 60, 60),
    "secondary_figure": (240, 130, 130),
    "animal": (60, 200, 90),
    "celestial": (80, 120, 255),
    "object": (240, 220, 70),
    "terrain": (140, 140, 140),
    "water": (70, 180, 255),
    "symbol": (220, 100, 220),
}

ANIMAL_WORDS = {"dog", "wolf", "lion", "horse", "crayfish", "beasts", "two beasts"}
CELESTIAL_WORDS = {"sun", "moon", "large star", "small stars", "stars", "halo", "cloud", "clouds"}
TERRAIN_WORDS = {"cliff", "mountain", "mountains", "path", "tower", "towers", "forest", "garden"}
WATER_WORDS = {"water", "river"}


def classify_symbol(s: str) -> str:
    s = s.lower()
    if s in ANIMAL_WORDS:
        return "animal"
    if s in CELESTIAL_WORDS:
        return "celestial"
    if s in TERRAIN_WORDS:
        return "terrain"
    if s in WATER_WORDS:
        return "water"
    return "object"


def draw_frame(draw: ImageDraw.ImageDraw) -> None:
    draw.rectangle([FRAME_MARGIN, FRAME_MARGIN, W - FRAME_MARGIN, H - FRAME_MARGIN], outline=COLORS["frame"], width=4)
    draw.rectangle([FRAME_MARGIN + 20, FRAME_MARGIN + 20, W - FRAME_MARGIN - 20, H - FRAME_MARGIN - 20], outline=COLORS["frame"], width=2)


def draw_main_figure(draw: ImageDraw.ImageDraw, box: List[int], color=None) -> None:
    color = color or COLORS["figure"]
    draw.rounded_rectangle(box, fill=color, radius=18)


def draw_circle(draw: ImageDraw.ImageDraw, center: Tuple[int, int], r: int, color) -> None:
    x, y = center
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def layout_single_figure_center(draw, spec):
    draw_main_figure(draw, [160, 170, 352, 560])


def layout_single_figure_bottom(draw, spec):
    draw_main_figure(draw, [170, 300, 342, 600])


def layout_double_figure(draw, spec):
    draw_main_figure(draw, [120, 220, 240, 560])
    draw_main_figure(draw, [272, 220, 392, 560], COLORS["secondary_figure"])


def layout_vehicle_center(draw, spec):
    draw_main_figure(draw, [170, 180, 342, 420])
    draw.rectangle([120, 420, 392, 560], fill=COLORS["object"])


def layout_symbol_center(draw, spec):
    draw_circle(draw, (256, 360), 110, COLORS["symbol"])


def layout_suspended_figure(draw, spec):
    draw.rectangle([180, 120, 332, 160], fill=COLORS["terrain"])
    draw_main_figure(draw, [220, 170, 292, 480])


def layout_figure_mount(draw, spec):
    draw_main_figure(draw, [180, 180, 320, 430])
    draw.polygon([(110, 430), (390, 430), (330, 560), (150, 560)], fill=COLORS["animal"])


def layout_triangle_top(draw, spec):
    draw.polygon([(256, 140), (140, 360), (372, 360)], fill=COLORS["figure"])
    draw_main_figure(draw, [120, 420, 220, 610], COLORS["secondary_figure"])
    draw_main_figure(draw, [292, 420, 392, 610], COLORS["secondary_figure"])


def layout_tower_scene(draw, spec):
    draw.rectangle([200, 180, 312, 590], fill=COLORS["terrain"])
    draw.polygon([(256, 90), (220, 180), (292, 180)], fill=COLORS["celestial"])


def layout_path_scene(draw, spec):
    draw_circle(draw, (256, 130), 60, COLORS["celestial"])
    draw.polygon([(230, 280), (282, 280), (340, 650), (172, 650)], fill=COLORS["terrain"])
    draw.rectangle([90, 240, 140, 520], fill=COLORS["terrain"])
    draw.rectangle([372, 240, 422, 520], fill=COLORS["terrain"])


def layout_top_bottom_scene(draw, spec):
    draw_main_figure(draw, [180, 100, 332, 280])
    draw.rectangle([120, 450, 180, 600], fill=COLORS["secondary_figure"])
    draw.rectangle([226, 430, 286, 600], fill=COLORS["secondary_figure"])
    draw.rectangle([332, 450, 392, 600], fill=COLORS["secondary_figure"])


def layout_multi_symbol(draw, spec):
    count = max(2, min(spec.get("count") or 2, 10))
    cols = 2 if count <= 4 else 3
    rows = (count + cols - 1) // cols
    x0, y0 = 120, 180
    cell_w, cell_h = 270 // cols, 360 // rows
    color = COLORS["symbol"]
    for i in range(count):
        r = i // cols
        c = i % cols
        cx = x0 + c * cell_w + cell_w // 2
        cy = y0 + r * cell_h + cell_h // 2
        draw_circle(draw, (cx, cy), min(cell_w, cell_h) // 4, color)


def layout_court_figure(draw, spec):
    draw_main_figure(draw, [170, 170, 342, 560])
    draw.rectangle([360, 250, 410, 500], fill=COLORS["symbol"])


def place_symbols(draw: ImageDraw.ImageDraw, spec: Dict[str, Any]) -> None:
    symbols = spec.get("symbols", [])
    for idx, sym in enumerate(symbols[:8]):
        kind = classify_symbol(sym)
        color = COLORS[kind]
        if kind == "celestial":
            positions = [(90, 110), (420, 110), (256, 100), (430, 170)]
            pos = positions[idx % len(positions)]
            draw_circle(draw, pos, 26 if "star" not in sym else 18, color)
        elif kind == "animal":
            positions = [(110, 560), (405, 560), (120, 470), (392, 470)]
            pos = positions[idx % len(positions)]
            draw_circle(draw, pos, 30, color)
        elif kind == "terrain":
            if sym in {"cliff", "mountain", "mountains"}:
                draw.polygon([(80, 650), (180, 560), (260, 650)], fill=color)
                draw.polygon([(252, 650), (340, 540), (430, 650)], fill=color)
            elif sym in {"tower", "towers"}:
                draw.rectangle([80, 280, 125, 560], fill=color)
                if sym == "towers":
                    draw.rectangle([387, 280, 432, 560], fill=color)
            elif sym == "path":
                draw.polygon([(240, 400), (272, 400), (330, 650), (182, 650)], fill=color)
        elif kind == "water":
            draw.rectangle([100, 590, 412, 670], fill=color)
        else:
            positions = [(100, 220), (410, 220), (100, 340), (410, 340), (100, 460), (410, 460)]
            x, y = positions[idx % len(positions)]
            draw.rounded_rectangle([x - 28, y - 28, x + 28, y + 28], fill=color, radius=8)


def render_one(spec: Dict[str, Any], out_path: Path) -> None:
    img = Image.new("RGB", (W, H), COLORS["background"])
    draw = ImageDraw.Draw(img)
    draw_frame(draw)

    layout_type = spec["layout_type"]
    if layout_type == "single_figure_center":
        layout_single_figure_center(draw, spec)
    elif layout_type == "single_figure_bottom":
        layout_single_figure_bottom(draw, spec)
    elif layout_type == "double_figure":
        layout_double_figure(draw, spec)
    elif layout_type == "vehicle_center":
        layout_vehicle_center(draw, spec)
    elif layout_type == "figure_animal":
        layout_single_figure_center(draw, spec)
    elif layout_type == "symbol_center":
        layout_symbol_center(draw, spec)
    elif layout_type == "suspended_figure":
        layout_suspended_figure(draw, spec)
    elif layout_type == "figure_mount":
        layout_figure_mount(draw, spec)
    elif layout_type == "triangle_top":
        layout_triangle_top(draw, spec)
    elif layout_type == "tower_scene":
        layout_tower_scene(draw, spec)
    elif layout_type == "path_scene":
        layout_path_scene(draw, spec)
    elif layout_type == "top_bottom_scene":
        layout_top_bottom_scene(draw, spec)
    elif layout_type == "multi_symbol":
        layout_multi_symbol(draw, spec)
    elif layout_type == "court_figure":
        layout_court_figure(draw, spec)
    else:
        layout_single_figure_center(draw, spec)

    place_symbols(draw, spec)
    img.save(out_path)


def render_layouts(specs: List[Dict[str, Any]]) -> None:
    ensure_dirs()
    for spec in specs:
        out_path = LAYOUTS_DIR / f"{spec['slug']}.png"
        render_one(spec, out_path)
    print(f"Rendered {len(specs)} layout images -> {LAYOUTS_DIR}")


# ============================================================
# Step 4. Build train.jsonl for ControlNet / Adapter
# ============================================================
def build_prompt(spec: Dict[str, Any], style_mode: str = "rws") -> str:
    card_name = spec["card_name"]
    arcana = spec["arcana"]
    suit = spec.get("suit")
    rank = spec.get("rank")
    mood = spec.get("mood", "")

    parts = ["tarot card", card_name, f"{arcana} arcana"]
    if suit:
        parts.append(f"{suit} suit")
    if rank:
        parts.append(f"{rank} rank")
    if mood:
        parts.append(mood)

    if style_mode == "pixel":
        parts.extend(["pixel art", "retro fantasy card", "ornate pixel border"])
    else:
        parts.extend(["rider waite smith style", "ornate border"])

    return ", ".join(parts)


def build_dataset_jsonl(specs: List[Dict[str, Any]], style_mode: str = "rws") -> List[Dict[str, str]]:
    ensure_dirs()
    records = []
    missing = []

    for spec in specs:
        slug = spec["slug"]
        image_path = IMAGES_DIR / f"{slug}.png"
        cond_path = LAYOUTS_DIR / f"{slug}.png"

        if not image_path.exists():
            missing.append(str(image_path))
            continue
        if not cond_path.exists():
            missing.append(str(cond_path))
            continue

        records.append({
            "image": str(image_path),
            "conditioning_image": str(cond_path),
            "text": build_prompt(spec, style_mode=style_mode),
        })

    with open(TRAIN_JSONL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records -> {TRAIN_JSONL_PATH}")
    if missing:
        print("Missing path sample:", missing[:10])
    return records


# ============================================================
# Step 5. Sanity check helpers
# ============================================================
def inspect_outputs(limit: int = 5) -> None:
    print("\n=== DATA SUMMARY ===")
    print("images:", len(list(IMAGES_DIR.glob("*.png"))))
    print("layouts:", len(list(LAYOUTS_DIR.glob("*.png"))))
    print("spec exists:", SPECS_PATH.exists())
    print("train.jsonl exists:", TRAIN_JSONL_PATH.exists())

    if TRAIN_JSONL_PATH.exists():
        print("\nFirst few train records:")
        with open(TRAIN_JSONL_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                print(line.strip())
                if i + 1 >= limit:
                    break


# ============================================================
# Main pipeline
# ============================================================
def run_pipeline(style_mode: str = "rws") -> None:
    _, cards = prepare_kaggle_tarot_images()
    specs = build_all_specs(cards)
    render_layouts(specs)
    build_dataset_jsonl(specs, style_mode=style_mode)
    inspect_outputs(limit=3)

    print("\n=== NEXT STEP ===")
    print("You can now train ControlNet / T2I-Adapter with:")
    print("- image: data/images/*.png")
    print("- conditioning_image: data/layouts/*.png")
    print("- text: data/train.jsonl")


if __name__ == "__main__":
    # style_mode: 'rws' or 'pixel'
    run_pipeline(style_mode="rws")
