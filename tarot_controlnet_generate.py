import os
import re
import json
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPTokenizer, CLIPTextModel


# ============================================================
# Tarot deck list
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
# Data structures
# ============================================================
@dataclass
class CardSpec:
    card_name: str
    main_figure: str
    symbols: List[str]
    pose: str
    mood: str
    composition: str
    background: str
    color_hints: List[str]
    count_hint: Optional[int] = None

    def to_token_sequence(self) -> str:
        tokens = [
            f"[CARD_{self._sanitize(self.card_name)}]",
            f"[FIGURE_{self._sanitize(self.main_figure)}]",
            f"[POSE_{self._sanitize(self.pose)}]",
            f"[MOOD_{self._sanitize(self.mood)}]",
            f"[COMP_{self._sanitize(self.composition)}]",
            f"[BG_{self._sanitize(self.background)}]",
        ]
        for s in self.symbols:
            tokens.append(f"[SYM_{self._sanitize(s)}]")
        for c in self.color_hints:
            tokens.append(f"[COLOR_{self._sanitize(c)}]")
        if self.count_hint is not None:
            tokens.append(f"[COUNT_{self.count_hint}]")
        return " ".join(tokens)

    @staticmethod
    def _sanitize(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "unknown"


# ============================================================
# Rule-based semantic parser
# ============================================================
class CardSemanticParser:
    def __init__(self):
        self.major_templates = self._build_major_templates()

    def parse(self, global_prompt: str, card_name: str) -> CardSpec:
        style_colors = self._extract_color_hints(global_prompt)
        if card_name in self.major_templates:
            spec = self.major_templates[card_name]
            return CardSpec(
                card_name=card_name,
                main_figure=spec["main_figure"],
                symbols=spec["symbols"],
                pose=spec["pose"],
                mood=spec["mood"],
                composition=spec["composition"],
                background=spec["background"],
                color_hints=style_colors,
                count_hint=spec.get("count_hint"),
            )
        return self._parse_minor_arcana(card_name, style_colors)

    def _extract_color_hints(self, prompt: str) -> List[str]:
        vocab = [
            "gold", "silver", "red", "blue", "green", "purple", "black", "white", "crimson",
            "emerald", "sapphire", "amber", "violet", "teal", "bronze", "ivory", "scarlet",
            "indigo", "obsidian", "pastel", "monochrome"
        ]
        prompt_l = prompt.lower()
        found = [c for c in vocab if c in prompt_l]
        return found if found else ["muted", "ornamental"]

    def _parse_minor_arcana(self, card_name: str, color_hints: List[str]) -> CardSpec:
        m = re.match(r"^(Ace|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Page|Knight|Queen|King) of (Wands|Cups|Swords|Pentacles)$", card_name)
        if not m:
            raise ValueError(f"Unknown tarot card: {card_name}")

        rank, suit = m.group(1), m.group(2)
        count_hint = self._rank_to_number(rank)
        main_figure, pose, mood, composition, background = self._minor_arcana_defaults(rank, suit)
        symbols = [suit[:-1].lower() if suit.endswith("s") else suit.lower()]
        if count_hint is not None:
            symbols += [suit[:-1].lower()] * max(count_hint - 1, 0)

        return CardSpec(
            card_name=card_name,
            main_figure=main_figure,
            symbols=symbols,
            pose=pose,
            mood=mood,
            composition=composition,
            background=background,
            color_hints=color_hints,
            count_hint=count_hint,
        )

    def _minor_arcana_defaults(self, rank: str, suit: str) -> Tuple[str, str, str, str, str]:
        suit_theme = {
            "Wands": ("traveler or guardian", "standing confidently", "energetic", "vertical and dynamic", "open landscape with fire motifs"),
            "Cups": ("dreamer or celebrant", "holding a vessel", "emotional", "balanced and flowing", "waterfront or moonlit scene"),
            "Swords": ("warrior or thinker", "facing tension", "intense", "sharp diagonal structure", "windy sky or battlefield"),
            "Pentacles": ("craftsperson or sovereign", "displaying an emblem", "grounded", "symmetrical and stable", "garden, city, or stone setting"),
        }
        base = list(suit_theme[suit])

        if rank in ["Page", "Knight", "Queen", "King"]:
            figure_map = {
                "Page": "youthful court figure",
                "Knight": "mounted or advancing knight",
                "Queen": "regal seated queen",
                "King": "commanding seated king",
            }
            pose_map = {
                "Page": "presenting the suit symbol",
                "Knight": "moving forward with purpose",
                "Queen": "seated in calm authority",
                "King": "enthroned with control",
            }
            mood_map = {
                "Page": "curious",
                "Knight": "determined",
                "Queen": "composed",
                "King": "authoritative",
            }
            base[0] = figure_map[rank]
            base[1] = pose_map[rank]
            base[2] = mood_map[rank]
        elif rank == "Ace":
            base[0] = f"single radiant {suit[:-1].lower()} emblem"
            base[1] = "floating or centrally displayed"
            base[2] = "pure potential"
            base[3] = "centered sacred composition"
        return tuple(base)

    def _rank_to_number(self, rank: str) -> Optional[int]:
        mapping = {
            "Ace": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
            "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9, "Ten": 10,
        }
        return mapping.get(rank)

    def _build_major_templates(self) -> Dict[str, Dict]:
        return {
            "The Fool": {
                "main_figure": "young traveler",
                "symbols": ["dog", "cliff", "sun"],
                "pose": "stepping forward carelessly",
                "mood": "innocent and adventurous",
                "composition": "centered figure with open space",
                "background": "mountaintop under bright sky",
            },
            "The Magician": {
                "main_figure": "ritual magician",
                "symbols": ["wand", "cup", "sword", "pentacle", "table", ],
                "pose": "one hand upward one downward",
                "mood": "focused and powerful",
                "composition": "frontal ceremonial portrait",
                "background": "altar with floral details",
            },
            "The High Priestess": {
                "main_figure": "mystic priestess",
                "symbols": ["moon", "scroll", "pillars", "veil", "pomegranate"],
                "pose": "seated in stillness",
                "mood": "mysterious and contemplative",
                "composition": "symmetrical throne composition",
                "background": "temple interior with veil",
            },
            "The Empress": {
                "main_figure": "fertility empress",
                "symbols": ["crown", "wheat", "venus symbol", "forest", "throne"],
                "pose": "seated in abundance",
                "mood": "nurturing and abundant",
                "composition": "lush centered portrait",
                "background": "garden with river and grain",
            },
            "The Emperor": {
                "main_figure": "sovereign emperor",
                "symbols": ["ram", "throne", "scepter", "orb", "mountains"],
                "pose": "seated firmly",
                "mood": "stable and authoritative",
                "composition": "rigid throne composition",
                "background": "stone throne before mountains",
            },
            "The Hierophant": {
                "main_figure": "spiritual teacher",
                "symbols": ["staff", "acolytes", "keys", "throne"],
                "pose": "giving blessing",
                "mood": "traditional and solemn",
                "composition": "formal frontal arrangement",
                "background": "cathedral or sanctum",
            },
            "The Lovers": {
                "main_figure": "pair of lovers",
                "symbols": ["angel", "tree", "serpent", "sun"],
                "pose": "standing together",
                "mood": "harmonious and fateful",
                "composition": "triangular composition",
                "background": "garden landscape",
            },
            "The Chariot": {
                "main_figure": "victorious charioteer",
                "symbols": ["chariot", "two beasts", "armor", "canopy"],
                "pose": "standing in chariot",
                "mood": "driven and controlled",
                "composition": "forward-facing dynamic symmetry",
                "background": "city gate or road",
            },
            "Strength": {
                "main_figure": "calm tamer",
                "symbols": ["lion"],
                "pose": "gently holding lion",
                "mood": "calm courage",
                "composition": "soft centered interaction",
                "background": "sunlit meadow",
            },
            "The Hermit": {
                "main_figure": "elder hermit",
                "symbols": ["lantern", "snowy mountain"],
                "pose": "standing alone with lantern",
                "mood": "wise and introspective",
                "composition": "solitary vertical figure",
                "background": "night mountain path",
            },
            "Wheel of Fortune": {
                "main_figure": "mystic wheel",
                "symbols": ["wheel", "sphinx", "serpent", "winged creatures"],
                "pose": "central sacred emblem",
                "mood": "cyclical and cosmic",
                "composition": "radial circular composition",
                "background": "clouded celestial sky",
            },
            "Justice": {
                "main_figure": "judge figure",
                "symbols": ["scales", "sword", "throne", "curtain"],
                "pose": "seated upright",
                "mood": "balanced and exact",
                "composition": "strict bilateral symmetry",
                "background": "court-like chamber",
            },
            "The Hanged Man": {
                "main_figure": "suspended figure",
                "symbols": ["tree", "halo", "crossed leg"],
                "pose": "hanging upside down",
                "mood": "surrendered and enlightened",
                "composition": "vertical suspended composition",
                "background": "quiet sacred grove",
            },
            "Death": {
                "main_figure": "armored rider or skeletal figure",
                "symbols": ["white rose flag", "horse", "fallen king", "sunrise"],
                "pose": "advancing inexorably",
                "mood": "transformative and grave",
                "composition": "processional composition",
                "background": "river and horizon",
            },
            "Temperance": {
                "main_figure": "angelic figure",
                "symbols": ["two cups", "water flow", "iris flowers", "path"],
                "pose": "pouring liquid between vessels",
                "mood": "balanced and healing",
                "composition": "calm centered figure",
                "background": "riverbank with sunrise",
            },
            "The Devil": {
                "main_figure": "horned dark figure",
                "symbols": ["chains", "two captives", "torch", "pedestal"],
                "pose": "looming above chained figures",
                "mood": "tempting and oppressive",
                "composition": "heavy triangular composition",
                "background": "shadowy cavern or void",
            },
            "The Tower": {
                "main_figure": "struck tower",
                "symbols": ["lightning", "crown", "falling figures", "fire"],
                "pose": "tower exploding under lightning",
                "mood": "catastrophic and sudden",
                "composition": "violent diagonal energy",
                "background": "storm sky",
            },
            "The Star": {
                "main_figure": "hopeful maiden",
                "symbols": ["large star", "seven small stars", "water", "bird"],
                "pose": "kneeling and pouring water",
                "mood": "hopeful and serene",
                "composition": "open celestial composition",
                "background": "night sky over pool",
            },
            "The Moon": {
                "main_figure": "moonlit path",
                "symbols": ["moon", "wolf", "dog", "crayfish", "towers"],
                "pose": "path leading into mystery",
                "mood": "dreamlike and uncertain",
                "composition": "deep receding perspective",
                "background": "night marshland",
            },
            "The Sun": {
                "main_figure": "radiant child or youth",
                "symbols": ["sunflowers", "sun", "wall", "horse"],
                "pose": "joyful forward movement",
                "mood": "bright and triumphant",
                "composition": "open glowing frontal scene",
                "background": "sunlit garden",
            },
            "Judgement": {
                "main_figure": "angel above awakening figures",
                "symbols": ["trumpet", "coffins", "mountains", "banner"],
                "pose": "calling souls upward",
                "mood": "awakening and redemptive",
                "composition": "vertical spiritual ascent",
                "background": "misty horizon and sky",
            },
            "The World": {
                "main_figure": "dancing cosmic figure",
                "symbols": ["wreath", "four creatures", "batons"],
                "pose": "floating within wreath",
                "mood": "complete and harmonious",
                "composition": "oval centered cosmic symmetry",
                "background": "celestial expanse",
            },
        }


# ============================================================
# Global deck style encoder
# ============================================================
class GlobalDeckStyleEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
        self.text_encoder.eval()

    @torch.no_grad()
    def encode(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        style_vec = outputs.last_hidden_state.mean(dim=1)
        return style_vec.squeeze(0)


# ============================================================
# Prompt builder
# ============================================================
# class PromptConditioner:
#     def __init__(self, render_mode: str = "illustration"):
#         self.render_mode = render_mode

#     def build_prompt(self, global_prompt: str, card_name: str, spec: CardSpec, z_deck: torch.Tensor) -> str:
#         style_phrases = self._deck_style_phrases(z_deck)
#         symbol_str = ", ".join(spec.symbols)
#         color_str = ", ".join(spec.color_hints)

#         if self.render_mode == "pixel":
#             quality_suffix = (
#                 "pixel art tarot card, retro 16-bit fantasy RPG style, clean pixel shapes, "
#                 "limited palette, crisp edges, sprite-like composition, no smooth shading, "
#                 "ornate pixel border, centered pixel-art layout"
#             )
#         else:
#             quality_suffix = (
#                 "highly detailed, cohesive tarot deck, ornate border, centered composition, mystical illustration"
#             )

#         prompt = (
#             f"Tarot card illustration, {card_name}. "
#             f"Global style: {global_prompt}. "
#             f"Shared deck style: {style_phrases}. "
#             f"Main figure: {spec.main_figure}. "
#             f"Symbols: {symbol_str}. "
#             f"Pose: {spec.pose}. "
#             f"Mood: {spec.mood}. "
#             f"Composition: {spec.composition}. "
#             f"Background: {spec.background}. "
#             f"Color palette: {color_str}. "
#             f"{quality_suffix}."
#         )
#         return prompt

#     def build_negative_prompt(self) -> str:
#         if self.render_mode == "pixel":
#             return (
#                 "blurry, low quality, text, watermark, logo, cropped, malformed hands, extra limbs, duplicate objects, "
#                 "bad anatomy, distorted face, photorealistic, 3d render, smooth gradients, painterly texture, "
#                 "soft brush strokes, realistic lighting, anti-aliased edges"
#             )
#         return (
#             "blurry, low quality, text, watermark, logo, cropped, malformed hands, extra limbs, duplicate objects, "
#             "bad anatomy, distorted face, photorealistic, 3d render, modern clothes"
#         )

#     def _deck_style_phrases(self, z_deck: torch.Tensor) -> str:
#         values = z_deck.detach().float().cpu()
#         seed_value = int(torch.abs(values[:8]).sum().item() * 1000) % 10_000_000
#         rng = random.Random(seed_value)

#         if self.render_mode == "pixel":
#             palette = rng.choice([
#                 "restricted jewel-tone palette",
#                 "dark fantasy 16-color palette",
#                 "warm retro console palette",
#                 "moonlit blue-purple palette",
#             ])
#             lighting = rng.choice([
#                 "tile-based light contrast",
#                 "simple highlight clusters",
#                 "dramatic sprite lighting",
#                 "retro dungeon glow",
#             ])
#             border = rng.choice([
#                 "golden pixel filigree border",
#                 "ornamental pixel frame",
#                 "celestial pixel border",
#                 "ancient rune pixel frame",
#             ])
#             texture = rng.choice([
#                 "clean sprite rendering",
#                 "crisp pixel clusters",
#                 "retro tactical RPG aesthetic",
#                 "SNES-inspired fantasy iconography",
#             ])
#             return f"{palette}, {lighting}, {border}, {texture}"

#         brush = rng.choice([
#             "oil-painted brushwork",
#             "ink-and-gouache texture",
#             "etched storybook lines",
#             "soft painterly gradients",
#         ])
#         lighting = rng.choice([
#             "dramatic rim lighting",
#             "moonlit glow",
#             "golden sacred illumination",
#             "dim mystical ambience",
#         ])
#         ornament = rng.choice([
#             "ornate golden filigree",
#             "celestial border motifs",
#             "antique engraved frame",
#             "floral arcane ornament",
#         ])
#         palette = rng.choice([
#             "harmonized muted palette",
#             "rich jewel tones",
#             "deep ceremonial colors",
#             "aged parchment accents",
#         ])
#         return f"{brush}, {lighting}, {ornament}, {palette}"



# class PromptConditioner:
#     def __init__(self, render_mode: str = "illustration"):
#         self.render_mode = render_mode

#     def build_prompt(self, global_prompt: str, card_name: str, spec: CardSpec, z_deck: torch.Tensor) -> str:
#         style_phrases = self._deck_style_phrases(z_deck)
#         symbol_str = ", ".join(spec.symbols)
#         color_str = ", ".join(spec.color_hints)

#         # 强化 main_figure，把人物身份放到最前面，并重复一次
#         figure_block = (
#             # f"The main figure is {spec.main_figure}. "
#             # f"The central character is clearly {spec.main_figure}. "
#             # f"{spec.main_figure} is prominently depicted, {spec.pose}. "
#              f"The main figure is {spec.main_figure}. "
#             f"The central character is clearly {spec.main_figure}. "
#             f"{spec.main_figure} is prominently depicted, {spec.pose}. "
#             f"A large central figure dominates the composition. "
#             f"The main subject occupies most of the card. "
#             f"The character is visually prominent and much larger than secondary symbols. "
#         )

#         if self.render_mode == "pixel":
#             quality_suffix = (
#                 "pixel art tarot card, retro 16-bit fantasy RPG style, "
#                 "clean pixel shapes, limited palette, crisp edges, "
#                 "sprite-like composition, ornate pixel border, "
#                 "the main character is visually clear and prominent"
#             )
#         else:
#             quality_suffix = (
#                 "highly detailed tarot card illustration, ornate border, centered composition, "
#                 "mystical illustration, the main character is visually clear, dominant, and prominent"
#             )

#         prompt = (
#             f"{spec.main_figure}, {spec.pose}. "   # 句首先强调一次
#             f"Tarot card illustration of {card_name}. "
#             f"{figure_block}"
#             f"Visible symbols include {symbol_str}. "
#             f"The mood is {spec.mood}. "
#             f"The scene composition is {spec.composition}. "
#             f"The background is {spec.background}. "
#             f"Global style: {global_prompt}. "
#             f"Shared deck style: {style_phrases}. "
#             f"Color palette: {color_str}. "
#             f"{quality_suffix}."
#         )
#         return prompt

#     def build_negative_prompt(self) -> str:
#         if self.render_mode == "pixel":
#             return (
#                 "blurry, low quality, text, watermark, logo, cropped, malformed hands, extra limbs, "
#                 "bad anatomy, distorted face, photorealistic, 3d render, smooth gradients, painterly texture, "
#                 "soft brush strokes, realistic lighting, anti-aliased edges, tiny character, unclear subject"
#             )
#         return (
#             "blurry, low quality, text, watermark, logo, cropped, malformed hands, extra limbs, "
#             "bad anatomy, distorted face, photorealistic, 3d render, modern clothes, tiny character, unclear subject"
#         )

#     def _deck_style_phrases(self, z_deck: torch.Tensor) -> str:
#         values = z_deck.detach().float().cpu()
#         seed_value = int(torch.abs(values[:8]).sum().item() * 1000) % 10_000_000
#         rng = random.Random(seed_value)

#         if self.render_mode == "pixel":
#             palette = rng.choice([
#                 "restricted jewel-tone palette",
#                 "dark fantasy 16-color palette",
#                 "warm retro console palette",
#                 "moonlit blue-purple palette",
#             ])
#             lighting = rng.choice([
#                 "tile-based light contrast",
#                 "simple highlight clusters",
#                 "dramatic sprite lighting",
#                 "retro dungeon glow",
#             ])
#             border = rng.choice([
#                 "golden pixel filigree border",
#                 "ornamental pixel frame",
#                 "celestial pixel border",
#                 "ancient rune pixel frame",
#             ])
#             texture = rng.choice([
#                 "clean sprite rendering",
#                 "crisp pixel clusters",
#                 "retro tactical RPG aesthetic",
#                 "SNES-inspired fantasy iconography",
#             ])
#             return f"{palette}, {lighting}, {border}, {texture}"

#         brush = rng.choice([
#             "oil-painted brushwork",
#             "ink-and-gouache texture",
#             "etched storybook lines",
#             "soft painterly gradients",
#         ])
#         lighting = rng.choice([
#             "dramatic rim lighting",
#             "moonlit glow",
#             "golden sacred illumination",
#             "dim mystical ambience",
#         ])
#         ornament = rng.choice([
#             "ornate golden filigree",
#             "celestial border motifs",
#             "antique engraved frame",
#             "floral arcane ornament",
#         ])
#         palette = rng.choice([
#             "harmonized muted palette",
#             "rich jewel tones",
#             "deep ceremonial colors",
#             "aged parchment accents",
#         ])
#         return f"{brush}, {lighting}, {ornament}, {palette}"

class PromptConditioner:
    def __init__(self, render_mode: str = "illustration"):
        self.render_mode = render_mode

    def build_prompt(self, global_prompt: str, card_name: str, spec: CardSpec, z_deck: torch.Tensor) -> str:
        # 只保留最关键的几个 symbol，避免超过 77 tokens
        key_symbols = self._pick_key_symbols(spec.symbols, max_symbols=3)
        symbol_str = ", ".join(key_symbols)

        # 压缩风格描述，避免 prompt 太长
        style_phrase = self._short_style_phrase(global_prompt, z_deck)

        if self.render_mode == "pixel":
            render_suffix = "pixel art, retro fantasy tarot card, ornate pixel border"
        else:
            render_suffix = "mystical tarot card, ornate border, oil painting"

        # 关键：把 main_figure 放前面，并且不要写成长句
        prompt = (
            f"{card_name}, {spec.main_figure}, {spec.pose}, "
            f"{symbol_str}, "
            f"{style_phrase}, "
            f"{render_suffix}"
        )

        return prompt

    def build_negative_prompt(self) -> str:
        if self.render_mode == "pixel":
            return (
                "blurry, low quality, text, watermark, logo, bad anatomy, "
                "extra limbs, distorted face, photorealistic, 3d render, smooth gradients"
            )
        return (
            "blurry, low quality, text, watermark, logo, bad anatomy, "
            "extra limbs, distorted face, photorealistic, 3d render"
        )

    def _pick_key_symbols(self, symbols: List[str], max_symbols: int = 3) -> List[str]:
        if not symbols:
            return []

        # 尽量优先保留更有辨识度的元素
        priority = [
            "sun", "moon", "star", "dog", "wolf", "lion", "horse",
            "cliff", "tower", "towers", "wheel", "wreath",
            "wand", "cup", "sword", "pentacle", "flag", "lantern"
        ]

        picked = []
        used = set()

        for p in priority:
            for s in symbols:
                if s == p and s not in used:
                    picked.append(s)
                    used.add(s)
                    if len(picked) >= max_symbols:
                        return picked

        for s in symbols:
            if s not in used:
                picked.append(s)
                used.add(s)
                if len(picked) >= max_symbols:
                    break

        return picked

    def _short_style_phrase(self, global_prompt: str, z_deck: torch.Tensor) -> str:
        prompt_l = global_prompt.lower()

        style_tokens = []

        # 从用户 prompt 里抽最短的高价值风格词
        vocab = [
            "dark", "mystical", "oil painting", "gold", "golden",
            "gothic", "fantasy", "vintage", "ornate",
            "pixel art", "retro", "dramatic", "moonlit"
        ]
        for v in vocab:
            if v in prompt_l:
                style_tokens.append(v)

        # 不够的话补一个简短 deck 风格
        if len(style_tokens) < 3:
            style_tokens.extend(self._deck_style_tokens(z_deck))

        # 去重，最多保留 4 个短词
        dedup = []
        for x in style_tokens:
            if x not in dedup:
                dedup.append(x)

        dedup = dedup[:4]
        return ", ".join(dedup) if dedup else "mystical, ornate"

    def _deck_style_tokens(self, z_deck: torch.Tensor) -> List[str]:
        values = z_deck.detach().float().cpu()
        seed_value = int(torch.abs(values[:8]).sum().item() * 1000) % 10_000_000
        rng = random.Random(seed_value)

        if self.render_mode == "pixel":
            return [
                rng.choice(["pixel art", "retro fantasy", "limited palette"]),
                rng.choice(["ornate pixel border", "clean pixel shapes"]),
            ]

        return [
            rng.choice(["oil painting", "etched lines", "painterly"]),
            rng.choice(["ornate", "golden", "moonlit"]),
        ]



# ============================================================
# ControlNet-based image generator
# ============================================================
class TarotImageGenerator:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_path: str = "outputs/controlnet_tarot_minimal/final",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.device = device

        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch_dtype,
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=False)

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()

        if device == "cuda" and hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: Image.Image,
        seed: int,
        num_inference_steps: int = 30,
        guidance_scale: float = 8.0,
        controlnet_conditioning_scale: float = 1.2,
        width: int = 512,
        height: int = 768,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        control_image = control_image.convert("RGB").resize((width, height))

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=width,
            height=height,
            generator=generator,
        )
        return out.images[0]


# ============================================================
# Full pipeline
# ============================================================
class TarotDeckPipeline:
    def __init__(
        self,
        style_encoder_model: str = "openai/clip-vit-large-patch14",
        diffusion_model: str = "runwayml/stable-diffusion-v1-5",
        controlnet_path: str = "outputs/controlnet_tarot_minimal/checkpoint-epoch-10/diffusion_pytorch_model.safetensors",
        layout_dir: str = "data/layouts",
        device: Optional[str] = None,
        render_mode: str = "illustration",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.layout_dir = layout_dir

        dtype = torch.float16 if device == "cuda" else torch.float32

        self.parser = CardSemanticParser()
        self.style_encoder = GlobalDeckStyleEncoder(style_encoder_model, device=device)
        self.conditioner = PromptConditioner(render_mode=render_mode)
        self.generator = TarotImageGenerator(
            model_id=diffusion_model,
            controlnet_path=controlnet_path,
            device=device,
            torch_dtype=dtype,
        )

    def _load_control_image(self, card_name: str) -> Image.Image:
        slug = self._safe_filename(card_name)
        layout_path = os.path.join(self.layout_dir, f"{slug}.png")
        if not os.path.exists(layout_path):
            raise FileNotFoundError(f"Control image not found: {layout_path}")
        return Image.open(layout_path).convert("RGB")

    def generate_card(
        self,
        global_prompt: str,
        card_name: str,
        out_dir: str,
        deck_seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        controlnet_conditioning_scale: float,
        width: int,
        height: int,
    ) -> Dict:
        os.makedirs(out_dir, exist_ok=True)

        z_deck = self.style_encoder.encode(global_prompt)
        spec = self.parser.parse(global_prompt, card_name)
        prompt = self.conditioner.build_prompt(global_prompt, card_name, spec, z_deck)
        negative_prompt = self.conditioner.build_negative_prompt()
        control_image = self._load_control_image(card_name)

        card_seed = self._stable_card_seed(deck_seed, card_name)
        image = self.generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=control_image,
            seed=card_seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=width,
            height=height,
        )

        base_name = self._safe_filename(card_name)
        image_path = os.path.join(out_dir, f"{base_name}.png")
        meta_path = os.path.join(out_dir, f"{base_name}.json")

        image.save(image_path)
        metadata = {
            "card_name": card_name,
            "global_prompt": global_prompt,
            "deck_seed": deck_seed,
            "card_seed": card_seed,
            "token_sequence": spec.to_token_sequence(),
            "spec": asdict(spec),
            "final_prompt": prompt,
            "negative_prompt": negative_prompt,
            "control_image_path": os.path.join(self.layout_dir, f"{base_name}.png"),
            "image_path": image_path,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return metadata

    def generate_deck(
        self,
        global_prompt: str,
        out_dir: str,
        cards: Optional[List[str]] = None,
        deck_seed: int = 1234,
        num_inference_steps: int = 30,
        guidance_scale: float = 8.0,
        controlnet_conditioning_scale: float = 1.2,
        width: int = 512,
        height: int = 768,
    ) -> List[Dict]:
        if cards is None:
            cards = TAROT_CARDS

        os.makedirs(out_dir, exist_ok=True)
        results = []
        for idx, card_name in enumerate(cards, start=1):
            print(f"[{idx}/{len(cards)}] Generating: {card_name}")
            result = self.generate_card(
                global_prompt=global_prompt,
                card_name=card_name,
                out_dir=out_dir,
                deck_seed=deck_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                width=width,
                height=height,
            )
            results.append(result)

        with open(os.path.join(out_dir, "deck_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return results

    @staticmethod
    def _safe_filename(name: str) -> str:
        name = name.lower().strip()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return name.strip("_")

    @staticmethod
    def _stable_card_seed(deck_seed: int, card_name: str) -> int:
        value = sum((i + 1) * ord(c) for i, c in enumerate(card_name))
        return (deck_seed * 1009 + value) % (2**31 - 1)


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tarot deck generation pipeline with a trained ControlNet."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Global tarot deck style prompt.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/tarot_deck_controlnet",
        help="Directory to save generated images and metadata.",
    )
    parser.add_argument(
        "--card",
        type=str,
        default=None,
        help="Generate only one card, e.g. 'The Fool'. If omitted, generate the full deck.",
    )
    parser.add_argument(
        "--deck_seed",
        type=int,
        default=1234,
        help="Shared deck seed for reproducible generation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Diffusion inference steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.2,
        help="How strongly ControlNet follows the layout image.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Image height.",
    )
    parser.add_argument(
        "--style_encoder_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Hugging Face model name for style encoding.",
    )
    parser.add_argument(
        "--diffusion_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base diffusion model.",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default="outputs/controlnet_tarot_minimal/final",
        help="Path to the trained ControlNet checkpoint.",
    )
    parser.add_argument(
        "--layout_dir",
        type=str,
        default="data/layouts",
        help="Directory containing control layout images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "cuda"],
        help="Force device. By default, auto-selects cuda if available.",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="illustration",
        choices=["illustration", "pixel"],
        help="Prompt rendering style.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = TarotDeckPipeline(
        style_encoder_model=args.style_encoder_model,
        diffusion_model=args.diffusion_model,
        controlnet_path=args.controlnet_path,
        layout_dir=args.layout_dir,
        device=args.device,
        render_mode=args.render_mode,
    )

    if args.card is not None:
        if args.card not in TAROT_CARDS:
            raise ValueError(f"Unknown card '{args.card}'.")

        result = pipeline.generate_card(
            global_prompt=args.prompt,
            card_name=args.card,
            out_dir=args.out_dir,
            deck_seed=args.deck_seed,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            width=args.width,
            height=args.height,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        results = pipeline.generate_deck(
            global_prompt=args.prompt,
            out_dir=args.out_dir,
            cards=TAROT_CARDS,
            deck_seed=args.deck_seed,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            width=args.width,
            height=args.height,
        )
        print(f"Generated {len(results)} cards into: {args.out_dir}")


if __name__ == "__main__":
    main()