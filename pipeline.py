import os
import re
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from urllib import request as urllib_request
from urllib import error as urllib_error

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline
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
        # suit_theme = {
        #     "Wands": ("traveler or guardian", "standing confidently", "energetic", "vertical and dynamic", "open landscape with fire motifs"),
        #     "Cups": ("dreamer or celebrant", "holding a vessel", "emotional", "balanced and flowing", "waterfront or moonlit scene"),
        #     "Swords": ("warrior or thinker", "facing tension", "intense", "sharp diagonal structure", "windy sky or battlefield"),
        #     "Pentacles": ("gold","raftsperson or sovereign", "displaying an emblem", "grounded", "symmetrical and stable", ),
        # }
        suit_theme = {
            "Wands": (
                "wooden staffs",
                "human figure",
                "staffs held or planted upright",
                "dry landscape",
                "traditional tarot style"
            ),
            "Cups": (
                "golden cups",
                "human figure",
                "cups held or arranged clearly",
                "water background",
                "traditional tarot style"
            ),
            "Swords": (
                "steel swords",
                "human figure",
                "swords crossed, raised, or aligned",
                "cloudy outdoor setting",
                "traditional tarot style"
            ),
            "Pentacles": (
                "gold pentacles with star emblem",
                "human figure",
                "pentacles held, stacked, or displayed",
                "earthy outdoor setting",
                "traditional tarot style"
            ),
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
# Optional vLLM / RunPod semantic parser
# ============================================================
class RunPodVLLMSemanticParser:
    def __init__(
        self,
        api_key: str,
        endpoint_id: str,
        model_name: str,
        fallback_parser: CardSemanticParser,
        polling_interval: float = 2.0,
        timeout_sec: int = 180,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.model_name = model_name
        self.fallback_parser = fallback_parser
        self.polling_interval = polling_interval
        self.timeout_sec = timeout_sec
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.run_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        self.status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status"

    def parse(self, global_prompt: str, card_name: str) -> CardSpec:
        fallback_spec = self.fallback_parser.parse(global_prompt, card_name)
        try:
            raw_spec = self._request_spec(global_prompt, card_name, fallback_spec)
            return self._coerce_to_card_spec(raw_spec, fallback_spec)
        except Exception as e:
            print(f"[vLLM spec] failed for {card_name}, fallback to rule parser: {e}")
            return fallback_spec

    def _request_spec(self, global_prompt: str, card_name: str, fallback_spec: CardSpec) -> Dict:
        prompt = self._build_prompt(global_prompt, card_name, fallback_spec)
        payload = {
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a tarot semantic planner. "
                            "Return exactly one complete valid JSON object. "
                            "No markdown. No explanation. "
                            "Do not omit required keys."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "sampling_params": {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
            }
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            self.run_url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=60) as resp:
                run_resp = json.loads(resp.read().decode("utf-8"))
        except urllib_error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"RunPod run HTTPError {e.code}: {body}") from e

        job_id = run_resp.get("id")
        if not job_id:
            raise RuntimeError(f"RunPod did not return job id: {run_resp}")

        result = self._poll_job(job_id)
        text = self._extract_text(result)
        return self._parse_json_text(text)

    def _poll_job(self, job_id: str) -> Dict:
        import time
        start = time.time()
        while True:
            req = urllib_request.Request(
                f"{self.status_url}/{job_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                method="GET",
            )
            try:
                with urllib_request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib_error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                raise RuntimeError(f"RunPod status HTTPError {e.code}: {body}") from e

            status = data.get("status")
            if status == "COMPLETED":
                return data
            if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                raise RuntimeError(f"RunPod job {job_id} failed: {data}")
            if time.time() - start > self.timeout_sec:
                raise TimeoutError(f"RunPod job timed out after {self.timeout_sec}s")
            time.sleep(self.polling_interval)

    def _extract_text(self, result_json: Dict) -> str:
        output = result_json.get("output")

        if isinstance(output, str):
            return output

        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                choices = first.get("choices")
                if isinstance(choices, list) and choices:
                    choice0 = choices[0]
                    if isinstance(choice0, dict):
                        if "tokens" in choice0 and isinstance(choice0["tokens"], list):
                            return "".join(choice0["tokens"])
                        if "text" in choice0 and isinstance(choice0["text"], str):
                            return choice0["text"]

        if isinstance(output, dict):
            for key in ["text", "response", "output", "generated_text", "content"]:
                if key in output and isinstance(output[key], str):
                    return output[key]

        return json.dumps(output, ensure_ascii=False)

    def _parse_json_text(self, text: str) -> Dict:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]
        return json.loads(cleaned)

    def _build_prompt(self, global_prompt: str, card_name: str, fallback_spec: CardSpec) -> str:
        fallback_payload = {
            "card_name": fallback_spec.card_name,
            "main_figure": fallback_spec.main_figure,
            "symbols": fallback_spec.symbols,
            "pose": fallback_spec.pose,
            "mood": fallback_spec.mood,
            "composition": fallback_spec.composition,
            "background": fallback_spec.background,
            "color_hints": fallback_spec.color_hints,
            "count_hint": fallback_spec.count_hint,
        }
        return f"""
Generate a structured tarot card specification for one card in a single coherent deck.

Global deck prompt:
{global_prompt}

Card name:
{card_name}

Return ONLY valid JSON with exactly these keys:
{{
  "card_name": "{card_name}",
  "main_figure": "string",
  "symbols": ["string", "string"],
  "pose": "string",
  "mood": "string",
  "composition": "string",
  "background": "string",
  "color_hints": ["string", "string"],
  "count_hint": 0
}}

Rules:
- Keep the card identity faithful to standard tarot meanings.
- Keep the description compatible with image generation.
- Use short concrete phrases, not paragraphs.
- For Major Arcana, set count_hint to null unless a repeated counted object is central.
- For Minor Arcana, set count_hint to the number of suit symbols when appropriate.
- color_hints should reflect the global deck prompt.
- symbols should list the key visible symbolic objects.
- no markdown fences
- no commentary

Fallback reference (keep semantics close if uncertain):
{json.dumps(fallback_payload, ensure_ascii=False)}
""".strip()

    def _coerce_to_card_spec(self, raw: Dict, fallback_spec: CardSpec) -> CardSpec:
        def ensure_str(value, fallback):
            if isinstance(value, str) and value.strip():
                return value.strip()
            return fallback

        def ensure_list(value, fallback):
            if isinstance(value, list):
                cleaned = [str(v).strip() for v in value if str(v).strip()]
                if cleaned:
                    return cleaned
            return fallback

        count_hint = raw.get("count_hint", fallback_spec.count_hint)
        if count_hint is not None:
            try:
                count_hint = int(count_hint)
            except Exception:
                count_hint = fallback_spec.count_hint

        return CardSpec(
            card_name=fallback_spec.card_name,
            main_figure=ensure_str(raw.get("main_figure"), fallback_spec.main_figure),
            symbols=ensure_list(raw.get("symbols"), fallback_spec.symbols),
            pose=ensure_str(raw.get("pose"), fallback_spec.pose),
            mood=ensure_str(raw.get("mood"), fallback_spec.mood),
            composition=ensure_str(raw.get("composition"), fallback_spec.composition),
            background=ensure_str(raw.get("background"), fallback_spec.background),
            color_hints=ensure_list(raw.get("color_hints"), fallback_spec.color_hints),
            count_hint=count_hint,
        )


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

class PromptConditioner:
    def __init__(self, render_mode: str = "illustration", style_preset: str = "impressionist"):
        self.render_mode = render_mode
        self.style_preset = style_preset

    def build_prompt(self, global_prompt: str, card_name: str, spec: CardSpec, z_deck: torch.Tensor) -> str:
        style_phrase = self._short_style_phrase(global_prompt, z_deck)
        figure_phrase = self._build_figure_phrase(spec)
        count_phrase = self._build_count_phrase(spec)
        symbol_phrase = self._build_symbol_phrase(spec)
        scene_phrase = self._build_scene_phrase(spec)
        frame_phrase = self._frame_phrase()

        prompt_parts = [
            card_name,
            figure_phrase,
            count_phrase,
            symbol_phrase,
            scene_phrase,
            style_phrase,
            frame_phrase,
        ]
        compact = ", ".join([p for p in prompt_parts if p])
        return self._truncate_for_clip(compact)

    def build_background_prompt(self, global_prompt: str, card_name: str, spec: CardSpec, z_deck: torch.Tensor) -> str:
        style_phrase = self._short_style_phrase(global_prompt, z_deck)
        figure_phrase = self._build_figure_phrase(spec)
        scene_phrase = self._build_scene_phrase(spec)
        frame_phrase = self._frame_phrase()
        anti_symbol_phrase = ""
        if card_name in MINOR_ARCANA and spec.count_hint is not None:
            noun = self._infer_suit_noun(card_name)
            anti_symbol_phrase = (
                f"background and main figure only, leave clean space for {spec.count_hint} {noun} emblems, "
                f"do not draw repeated {noun}s"
            )
        prompt_parts = [
            card_name,
            figure_phrase,
            anti_symbol_phrase,
            scene_phrase,
            style_phrase,
            frame_phrase,
        ]
        compact = ", ".join([p for p in prompt_parts if p])
        return self._truncate_for_clip(compact)

    def build_symbol_prompt(self, card_name: str, global_prompt: str, z_deck: torch.Tensor) -> str:
        noun = self._infer_suit_noun(card_name)
        style_phrase = self._short_style_phrase(global_prompt, z_deck)
        prompt = (
            f"single {noun} emblem, centered, isolated, ornate tarot icon, minimal background, "
            f"clean silhouette, no frame, no duplicate objects, {style_phrase}"
        )
        return self._truncate_for_clip(prompt, max_words=30)

    def build_negative_prompt(self) -> str:
        common = (
            "blurry, low quality, text, watermark, logo, signature, cropped, bad anatomy, "
            "extra limbs, extra fingers, distorted face, wrong number of objects, tiny subject, "
            "off-center subject, collage, split panels, extra objects, overlapping objects, hard to count"
        )
        painterly = "photorealistic, 3d render, glossy cgi, modern clothes, flat vector art"
        return f"{common}, {painterly}"

    def _build_figure_phrase(self, spec: CardSpec) -> str:
        return f"{spec.main_figure}, {spec.pose}, {spec.mood}"

    def _build_count_phrase(self, spec: CardSpec) -> str:
        if spec.count_hint is None:
            return ""
        noun = self._infer_suit_noun(spec.card_name)
        if spec.count_hint == 1:
            return f"exactly one prominent {noun}, single clearly visible symbol"
        return f"exactly {spec.count_hint} visible {noun}s, non-overlapping, easy to count"

    def _build_symbol_phrase(self, spec: CardSpec) -> str:
        key_symbols = self._pick_key_symbols(spec.symbols, max_symbols=3)
        if not key_symbols:
            return ""
        if spec.count_hint is not None:
            key_symbols = [s for s in key_symbols if s != self._infer_suit_noun(spec.card_name)]
        if not key_symbols:
            return ""
        return "symbols: " + ", ".join(key_symbols)

    def _build_scene_phrase(self, spec: CardSpec) -> str:
        color_str = ", ".join(spec.color_hints[:2]) if spec.color_hints else "harmonized palette"
        return f"{spec.composition}, {spec.background}, {color_str} palette"

    def _frame_phrase(self) -> str:
        return "pixel tarot frame" if self.render_mode == "pixel" else "ornate tarot border, vertical tarot card"

    def _pick_key_symbols(self, symbols: List[str], max_symbols: int = 3) -> List[str]:
        unique = []
        for s in symbols:
            if s not in unique:
                unique.append(s)
        priority = [
            "sun", "moon", "star", "dog", "wolf", "lion", "horse", "cliff", "tower",
            "wheel", "wreath", "wand", "cup", "sword", "pentacle", "lantern", "flag"
        ]
        picked = [s for s in priority if s in unique][:max_symbols]
        if len(picked) < max_symbols:
            for s in unique:
                if s not in picked:
                    picked.append(s)
                    if len(picked) >= max_symbols:
                        break
        return picked

    def _short_style_phrase(self, global_prompt: str, z_deck: torch.Tensor) -> str:
        prompt_l = global_prompt.lower()
        if self.render_mode == "pixel":
            return "pixel art, limited palette, crisp sprite shading"
        if self.style_preset == "impressionist" or "impression" in prompt_l:
            color_phrase = self._extract_palette_hint(prompt_l)
            return f"impressionist oil painting, visible brushstrokes, luminous atmosphere, {color_phrase}"
        if "gothic" in prompt_l:
            return "gothic tarot illustration, dramatic shadow, antique ornament"
        if "art nouveau" in prompt_l:
            return "art nouveau illustration, elegant floral ornament, flowing lines"
        if "watercolor" in prompt_l:
            return "watercolor illustration, soft pigment bloom, delicate paper texture"
        deck_tokens = self._deck_style_tokens(z_deck)
        return ", ".join(deck_tokens[:3]) if deck_tokens else "painterly tarot illustration"

    def _extract_palette_hint(self, prompt_l: str) -> str:
        palette_map = [
            ("blue", "blue palette"),
            ("gold", "gold accents"),
            ("silver", "silver accents"),
            ("violet", "violet palette"),
            ("purple", "purple palette"),
            ("emerald", "emerald palette"),
            ("red", "warm red palette"),
            ("amber", "amber glow"),
            ("moonlit", "soft moonlit glow"),
        ]
        found = [label for key, label in palette_map if key in prompt_l]
        return found[0] if found else "cohesive painted palette"

    def _deck_style_tokens(self, z_deck: torch.Tensor) -> List[str]:
        values = z_deck.detach().float().cpu()
        seed_value = int(torch.abs(values[:8]).sum().item() * 1000) % 10_000_000
        rng = random.Random(seed_value)
        return [
            rng.choice(["ornate tarot border", "decorative celestial frame", "antique tarot frame"]),
            rng.choice(["cohesive painted palette", "consistent deck lighting", "storybook atmosphere"]),
            rng.choice(["mystical illustration", "painted tarot scene", "antique print feel"]),
        ]

    def _truncate_for_clip(self, prompt: str, max_words: int = 34) -> str:
        words = prompt.split()
        return prompt if len(words) <= max_words else " ".join(words[:max_words])

    def _infer_suit_noun(self, card_name: str) -> str:
        for suit, noun in {
            "Wands": "wand",
            "Cups": "cup",
            "Swords": "sword",
            "Pentacles": "pentacle",
        }.items():
            if suit in card_name:
                return noun
        return "symbol"


class TarotImageGenerator:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_path: str = "outputs/controlnet_tarot_minimal/checkpoint-epoch-20",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        lora_path: Optional[str] = None,
        lora_scale: float = 0.85,
        symbol_cache_dir: str = "outputs/symbol_cache",
    ):
        self.device = device
        self.lora_scale = lora_scale
        self.symbol_cache_dir = symbol_cache_dir
        os.makedirs(self.symbol_cache_dir, exist_ok=True)

        # if not os.path.isdir(controlnet_path):
        #     raise FileNotFoundError(
        #         f"ControlNet path must be a local diffusers directory containing config.json, got: {controlnet_path}"
        #     )
        # cfg_path = os.path.join(controlnet_path, "config.json")
        # if not os.path.exists(cfg_path):
        #     raise FileNotFoundError(
        #         f"Missing config.json in ControlNet directory: {controlnet_path}"
        #     )
        is_local_dir = os.path.isdir(controlnet_path)
        if is_local_dir:
            config_path = os.path.join(controlnet_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"Local ControlNet directory must contain config.json, got: {controlnet_path}"
                )

            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
        else:
            # 当成 Hugging Face repo id
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_path,
                torch_dtype=torch_dtype,
            )

        # self.controlnet = ControlNetModel.from_pretrained(
        #     controlnet_path,
        #     torch_dtype=torch_dtype,
        #     local_files_only=True,
        # )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            local_files_only=False,
        ).to(device)

        self.symbol_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            local_files_only=False,
        ).to(device)

        self.pipe.set_progress_bar_config(disable=False)
        self.symbol_pipe.set_progress_bar_config(disable=False)

        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if hasattr(self.symbol_pipe, "enable_attention_slicing"):
            self.symbol_pipe.enable_attention_slicing()

        self.has_lora = False
        if lora_path:
            try:
                self.pipe.load_lora_weights(lora_path)
                self.symbol_pipe.load_lora_weights(lora_path)
                self.has_lora = True
                print(f"[LoRA] loaded from: {lora_path}")
            except Exception as e:
                print(f"[LoRA] failed to load {lora_path}: {e}")

        if device == "cuda":
            if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            if hasattr(self.symbol_pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self.symbol_pipe.enable_xformers_memory_efficient_attention()
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
        lora_scale: Optional[float] = None,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        control_image = control_image.convert("RGB").resize((width, height))

        cross_attention_kwargs = None
        if self.has_lora:
            scale = self.lora_scale if lora_scale is None else lora_scale
            cross_attention_kwargs = {"scale": float(scale)}

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
            cross_attention_kwargs=cross_attention_kwargs,
        )
        return out.images[0]

    def _symbol_cache_path(self, suit_name: str, prompt: str, seed: int) -> str:
        key = f"{suit_name}_{abs(hash(prompt)) % 10_000_000}_{seed % 100000}"
        return os.path.join(self.symbol_cache_dir, f"{key}.png")

    def _remove_near_white_bg(self, image: Image.Image, white_thresh: int = 245) -> Image.Image:
        rgba = image.convert("RGBA")
        arr = np.array(rgba)
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        bg_mask = (r >= white_thresh) & (g >= white_thresh) & (b >= white_thresh)
        arr[bg_mask, 3] = 0
        alpha = arr[..., 3]
        nonzero = np.argwhere(alpha > 8)
        if len(nonzero) == 0:
            return rgba
        y0, x0 = nonzero.min(axis=0)
        y1, x1 = nonzero.max(axis=0)
        pad = 8
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(arr.shape[0] - 1, y1 + pad)
        x1 = min(arr.shape[1] - 1, x1 + pad)
        return Image.fromarray(arr[y0:y1 + 1, x0:x1 + 1], mode="RGBA")

    @torch.no_grad()
    def generate_symbol(
        self,
        suit_name: str,
        prompt: str,
        seed: int,
        negative_prompt: str = (
            "multiple objects, full scene, person, hands, landscape, border, card frame, text, "
            "background clutter, table"
        ),
        width: int = 256,
        height: int = 256,
        num_inference_steps: int = 24,
        guidance_scale: float = 7.5,
        lora_scale: Optional[float] = None,
        use_cache: bool = True,
    ) -> Image.Image:
        cache_path = self._symbol_cache_path(suit_name, prompt, seed)
        if use_cache and os.path.exists(cache_path):
            return Image.open(cache_path).convert("RGBA")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        cross_attention_kwargs = None
        if self.has_lora:
            scale = self.lora_scale if lora_scale is None else lora_scale
            cross_attention_kwargs = {"scale": float(scale)}

        out = self.symbol_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        image = self._remove_near_white_bg(out.images[0])
        if use_cache:
            image.save(cache_path)
        return image

    @torch.no_grad()
    def generate_symbol_candidates(
        self,
        suit_name: str,
        prompt: str,
        base_seed: int,
        negative_prompt: str = (
            "multiple objects, full scene, person, hands, landscape, border, card frame, text, "
            "background clutter, table"
        ),
        width: int = 256,
        height: int = 256,
        num_inference_steps: int = 24,
        guidance_scale: float = 7.5,
        lora_scale: Optional[float] = None,
        num_candidates: int = 4,
        use_cache: bool = True,
    ) -> List[Image.Image]:
        images = []
        for i in range(num_candidates):
            seed = base_seed + i * 97
            img = self.generate_symbol(
                suit_name=suit_name,
                prompt=prompt,
                seed=seed,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                lora_scale=lora_scale,
                use_cache=use_cache,
            )
            images.append(img)
        return images


class SymbolPrototypeGenerator:
    def __init__(
        self,
        image_generator: TarotImageGenerator,
        library_root: str = "outputs/symbol_library",
        warmup_symbols_per_suit: int = 8,
        candidate_batch_size: int = 8,
        store_top_k: int = 3,
        final_select_top_k: int = 3,
    ):
        self.image_generator = image_generator
        self.library_root = library_root
        self.warmup_symbols_per_suit = max(1, warmup_symbols_per_suit)
        self.candidate_batch_size = max(1, candidate_batch_size)
        self.store_top_k = max(1, store_top_k)
        self.final_select_top_k = max(1, final_select_top_k)
        os.makedirs(self.library_root, exist_ok=True)
        for suit in ["cup", "sword", "wand", "pentacle"]:
            os.makedirs(os.path.join(self.library_root, suit), exist_ok=True)

    def _library_dir(self, suit_name: str) -> str:
        return os.path.join(self.library_root, suit_name)

    def _list_library_symbols(self, suit_name: str) -> List[str]:
        d = self._library_dir(suit_name)
        return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith('.png')]

    def _target_aspect(self, suit_name: str) -> float:
        if suit_name == "sword":
            return 0.35
        if suit_name == "wand":
            return 0.45
        if suit_name == "cup":
            return 0.75
        return 1.0

    def _score_symbol_candidate(self, image: Image.Image, suit_name: str) -> float:
        arr = np.array(image.convert("RGBA"))
        alpha = arr[..., 3]
        fg = alpha > 12
        if fg.sum() < 20:
            return -1e9

        fg_ratio = float(fg.mean())
        area_score = 1.0 - min(abs(fg_ratio - 0.24) / 0.24, 1.0)

        coords = np.argwhere(fg)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        h = max(1, y1 - y0 + 1)
        w = max(1, x1 - x0 + 1)
        H, W = alpha.shape

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        center_dist = ((cx - W / 2) ** 2 + (cy - H / 2) ** 2) ** 0.5
        max_dist = ((W / 2) ** 2 + (H / 2) ** 2) ** 0.5
        center_score = 1.0 - center_dist / max_dist

        aspect = w / max(h, 1)
        target_aspect = self._target_aspect(suit_name)
        aspect_score = 1.0 - min(abs(aspect - target_aspect) / max(target_aspect, 1e-6), 1.0)

        edge_pixels = fg[0, :].sum() + fg[-1, :].sum() + fg[:, 0].sum() + fg[:, -1].sum()
        edge_ratio = float(edge_pixels) / max(float(fg.sum()), 1.0)
        edge_score = 1.0 - min(edge_ratio * 6.0, 1.0)

        bbox_area = h * w
        compactness = float(fg.sum()) / max(float(bbox_area), 1.0)
        compactness_score = min(compactness / 0.55, 1.0)

        score = (
            0.24 * area_score
            + 0.24 * center_score
            + 0.22 * aspect_score
            + 0.15 * edge_score
            + 0.15 * compactness_score
        )
        return float(score)

    def _infer_target_color(self, global_prompt: str) -> Tuple[int, int, int]:
        prompt_l = global_prompt.lower()
        if "gold" in prompt_l:
            return (212, 175, 55)
        if "silver" in prompt_l:
            return (205, 210, 220)
        if "blue" in prompt_l:
            return (90, 130, 210)
        if "red" in prompt_l or "crimson" in prompt_l:
            return (180, 70, 70)
        if "emerald" in prompt_l or "green" in prompt_l:
            return (70, 150, 110)
        if "purple" in prompt_l or "violet" in prompt_l:
            return (135, 95, 170)
        return (190, 160, 110)

    def _mean_foreground_color(self, image: Image.Image) -> np.ndarray:
        arr = np.array(image.convert("RGBA")).astype(np.float32)
        alpha = arr[..., 3] / 255.0
        mask = alpha > 0.05
        if not mask.any():
            return np.array([160.0, 160.0, 160.0], dtype=np.float32)
        rgb = arr[..., :3][mask]
        return rgb.mean(axis=0)

    def _color_match_score(self, image: Image.Image, global_prompt: str) -> float:
        target = np.array(self._infer_target_color(global_prompt), dtype=np.float32)
        mean_rgb = self._mean_foreground_color(image)
        dist = float(np.linalg.norm(mean_rgb - target))
        return 1.0 - min(dist / 255.0, 1.0)

    def _adapt_symbol_style(self, symbol: Image.Image, global_prompt: str, alpha_scale: float = 0.92) -> Image.Image:
        rgba = symbol.convert("RGBA")
        arr = np.array(rgba).astype(np.float32)
        target = np.array(self._infer_target_color(global_prompt), dtype=np.float32)
        alpha = arr[..., 3:4] / 255.0
        rgb = arr[..., :3]
        fg_mask = alpha > 0.05
        if fg_mask.any():
            rgb[fg_mask[..., 0]] = 0.68 * rgb[fg_mask[..., 0]] + 0.32 * target
        arr[..., :3] = np.clip(rgb, 0, 255)
        arr[..., 3] = np.clip(arr[..., 3] * alpha_scale, 0, 255)
        out = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        out = ImageEnhance.Color(out).enhance(0.92)
        out = ImageEnhance.Contrast(out).enhance(1.05)
        return out

    def _save_to_library(self, suit_name: str, image: Image.Image, seed: int, score: Optional[float] = None) -> str:
        d = self._library_dir(suit_name)
        idx = len(self._list_library_symbols(suit_name))
        suffix = f"_{score:.3f}" if score is not None else ""
        path = os.path.join(d, f"{suit_name}_{idx:03d}_{seed % 100000}{suffix}.png")
        image.save(path)
        return path

    def _bootstrap_library_if_needed(
        self,
        suit_name: str,
        symbol_prompt: str,
        base_seed: int,
        lora_scale: Optional[float],
    ) -> None:
        current = self._list_library_symbols(suit_name)
        if len(current) >= self.warmup_symbols_per_suit:
            return

        while len(current) < self.warmup_symbols_per_suit:
            candidates = self.image_generator.generate_symbol_candidates(
                suit_name=suit_name,
                prompt=symbol_prompt,
                base_seed=base_seed + len(current) * 1009,
                lora_scale=lora_scale,
                num_candidates=self.candidate_batch_size,
                use_cache=True,
            )
            scored = sorted(
                [(self._score_symbol_candidate(img, suit_name), img) for img in candidates],
                key=lambda x: x[0],
                reverse=True,
            )
            keep = scored[: min(self.store_top_k, len(scored))]
            for score, img in keep:
                if len(current) >= self.warmup_symbols_per_suit:
                    break
                self._save_to_library(suit_name, img, base_seed + len(current) * 17, score)
                current = self._list_library_symbols(suit_name)
            if not keep:
                break

    def _select_from_library(
        self,
        suit_name: str,
        global_prompt: str,
        card_name: str,
        seed: int,
    ) -> Optional[Image.Image]:
        files = self._list_library_symbols(suit_name)
        if not files:
            return None

        scored = []
        for path in files:
            img = Image.open(path).convert("RGBA")
            base_score = self._score_symbol_candidate(img, suit_name)
            color_score = self._color_match_score(img, global_prompt)
            total = 0.72 * base_score + 0.28 * color_score
            scored.append((total, path))

        scored.sort(key=lambda x: x[0], reverse=True)
        topk = scored[: min(self.final_select_top_k, len(scored))]
        rng = random.Random(abs(hash((suit_name, global_prompt, card_name, seed))))
        chosen_path = rng.choice([p for _, p in topk])
        return Image.open(chosen_path).convert("RGBA")

    def generate_symbol(
        self,
        card_name: str,
        global_prompt: str,
        z_deck: torch.Tensor,
        conditioner,
        seed: int,
        lora_scale: Optional[float] = None,
        num_candidates: Optional[int] = None,
    ) -> Image.Image:
        suit_name = conditioner._infer_suit_noun(card_name)
        symbol_prompt = conditioner.build_symbol_prompt(
            card_name=card_name,
            global_prompt=global_prompt,
            z_deck=z_deck,
        )
        if num_candidates is not None:
            self.candidate_batch_size = max(1, int(num_candidates))

        self._bootstrap_library_if_needed(
            suit_name=suit_name,
            symbol_prompt=symbol_prompt,
            base_seed=seed,
            lora_scale=lora_scale,
        )

        symbol = self._select_from_library(suit_name, global_prompt, card_name, seed)
        if symbol is None:
            candidates = self.image_generator.generate_symbol_candidates(
                suit_name=suit_name,
                prompt=symbol_prompt,
                base_seed=seed,
                lora_scale=lora_scale,
                num_candidates=self.candidate_batch_size,
                use_cache=True,
            )
            scored = sorted(
                [(self._score_symbol_candidate(img, suit_name), img) for img in candidates],
                key=lambda x: x[0],
                reverse=True,
            )
            if not scored:
                fallback = self.image_generator.generate_symbol(
                    suit_name=suit_name,
                    prompt=symbol_prompt,
                    seed=seed,
                    lora_scale=lora_scale,
                    use_cache=True,
                )
                symbol = fallback
                self._save_to_library(suit_name, fallback, seed)
            else:
                symbol = scored[0][1]
                self._save_to_library(suit_name, symbol, seed, scored[0][0])

        return self._adapt_symbol_style(symbol, global_prompt)


class SymbolPlacer:
    def place_symbols(
        self,
        background: Image.Image,
        symbol: Image.Image,
        slots: List[Tuple[int, int, int, int]],
        alpha: float = 0.90,
    ) -> Image.Image:
        canvas = background.convert("RGBA")
        for (x, y, w, h) in slots:
            s = symbol.copy().resize((w, h), Image.LANCZOS)
            if alpha < 1.0:
                a = s.getchannel("A")
                a = a.point(lambda p: int(p * alpha))
                s.putalpha(a)
            canvas.alpha_composite(s, (x, y))
        return canvas


class Harmonizer:
    def harmonize(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        img = ImageEnhance.Color(img).enhance(0.94)
        img = ImageEnhance.Contrast(img).enhance(1.06)
        img = ImageEnhance.Sharpness(img).enhance(1.02)
        return img.filter(ImageFilter.SMOOTH_MORE)


class TarotDeckPipeline:
    def __init__(
        self,
        style_encoder_model: str = "openai/clip-vit-large-patch14",
        diffusion_model: str = "runwayml/stable-diffusion-v1-5",
        controlnet_path: str = "outputs/controlnet_tarot_minimal/checkpoint-epoch-20",
        layout_dir: str = "data/layouts",
        device: Optional[str] = None,
        render_mode: str = "illustration",
        lora_path: Optional[str] = None,
        base_lora_scale: float = 0.85,
        style_preset: str = "impressionist",
        symbol_cache_dir: str = "outputs/symbol_cache",
        symbol_library_dir: str = "outputs/symbol_library",
        warmup_symbols_per_suit: int = 8,
        symbol_candidates: int = 8,
        store_top_k: int = 3,
        final_select_top_k: int = 3,
        spec_source: str = "rule",
        runpod_api_key: Optional[str] = None,
        runpod_endpoint_id: Optional[str] = None,
        vllm_model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        vllm_timeout_sec: int = 180,
        vllm_max_tokens: int = 256,
        vllm_temperature: float = 0.2,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.layout_dir = layout_dir
        self.symbol_candidates = symbol_candidates
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.rule_parser = CardSemanticParser()
        if spec_source == "vllm":
            if not runpod_api_key or not runpod_endpoint_id:
                raise ValueError("spec_source='vllm' requires --runpod_api_key and --runpod_endpoint_id")
            self.parser = RunPodVLLMSemanticParser(
                api_key=runpod_api_key,
                endpoint_id=runpod_endpoint_id,
                model_name=vllm_model_name,
                fallback_parser=self.rule_parser,
                timeout_sec=vllm_timeout_sec,
                max_tokens=vllm_max_tokens,
                temperature=vllm_temperature,
            )
        else:
            self.parser = self.rule_parser
        self.style_encoder = GlobalDeckStyleEncoder(style_encoder_model, device=device)
        self.conditioner = PromptConditioner(render_mode=render_mode, style_preset=style_preset)
        self.generator = TarotImageGenerator(
            model_id=diffusion_model,
            controlnet_path=controlnet_path,
            device=device,
            torch_dtype=dtype,
            lora_path=lora_path,
            lora_scale=base_lora_scale,
            symbol_cache_dir=symbol_cache_dir,
        )
        self.symbol_generator = SymbolPrototypeGenerator(
            self.generator,
            library_root=symbol_library_dir,
            warmup_symbols_per_suit=warmup_symbols_per_suit,
            candidate_batch_size=symbol_candidates,
            store_top_k=store_top_k,
            final_select_top_k=final_select_top_k,
        )
        self.symbol_placer = SymbolPlacer()
        self.harmonizer = Harmonizer()
        self.base_lora_scale = base_lora_scale

    def _load_control_image(self, card_name: str) -> Image.Image:
        slug = self._safe_filename(card_name)
        layout_path = os.path.join(self.layout_dir, f"{slug}.png")
        if not os.path.exists(layout_path):
            raise FileNotFoundError(f"Control image not found: {layout_path}")
        return Image.open(layout_path).convert("RGB")

    def _get_symbol_slots(self, card_name: str, width: int, height: int, count_hint: Optional[int]) -> List[Tuple[int, int, int, int]]:
        if count_hint is None or count_hint <= 0 or count_hint > 10:
            return []
        slot_w = int(width * 0.16)
        slot_h = int(slot_w * 1.15)
        top_y = int(height * 0.16)
        bottom_y = int(height * 0.62)
        if count_hint == 1:
            return [(width // 2 - slot_w // 2, int(height * 0.18), slot_w, slot_h)]
        n_top = (count_hint + 1) // 2
        n_bottom = count_hint - n_top
        def distribute(n: int, y: int) -> List[Tuple[int, int, int, int]]:
            if n <= 0:
                return []
            xs = []
            margin = int(width * 0.12)
            usable = width - 2 * margin - slot_w
            if n == 1:
                xs = [width // 2 - slot_w // 2]
            else:
                for i in range(n):
                    x = margin + int(i * usable / max(n - 1, 1))
                    xs.append(x)
            return [(x, y, slot_w, slot_h) for x in xs]
        return (distribute(n_top, top_y) + distribute(n_bottom, bottom_y))[:count_hint]

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
        lora_scale = self._style_scale_from_z_deck(z_deck)
        negative_prompt = self.conditioner.build_negative_prompt()
        control_image = self._load_control_image(card_name)
        card_seed = self._stable_card_seed(deck_seed, card_name)
        base_name = self._safe_filename(card_name)
        image_path = os.path.join(out_dir, f"{base_name}.png")
        meta_path = os.path.join(out_dir, f"{base_name}.json")

        if card_name in MAJOR_ARCANA:
            prompt = self.conditioner.build_prompt(global_prompt, card_name, spec, z_deck)
            final_image = self.generator.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=control_image,
                seed=card_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                width=width,
                height=height,
                lora_scale=lora_scale,
            )
            final_prompt = prompt
            branch = "major_direct"
        else:
            bg_prompt = self.conditioner.build_background_prompt(global_prompt, card_name, spec, z_deck)
            background = self.generator.generate(
                prompt=bg_prompt,
                negative_prompt=negative_prompt,
                control_image=control_image,
                seed=card_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                width=width,
                height=height,
                lora_scale=lora_scale,
            )
            if spec.count_hint is not None and 1 <= spec.count_hint <= 10:
                symbol = self.symbol_generator.generate_symbol(
                    card_name=card_name,
                    global_prompt=global_prompt,
                    z_deck=z_deck,
                    conditioner=self.conditioner,
                    seed=card_seed + 999,
                    lora_scale=lora_scale,
                    num_candidates=self.symbol_candidates,
                )
                slots = self._get_symbol_slots(card_name, width, height, spec.count_hint)
                composed = self.symbol_placer.place_symbols(background, symbol, slots)
                final_image = self.harmonizer.harmonize(composed)
                final_prompt = bg_prompt
                branch = "minor_background_plus_symbol_library"
            else:
                prompt = self.conditioner.build_prompt(global_prompt, card_name, spec, z_deck)
                final_image = self.generator.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    control_image=control_image,
                    seed=card_seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    width=width,
                    height=height,
                    lora_scale=lora_scale,
                )
                final_prompt = prompt
                branch = "minor_fallback_direct"

        final_image.save(image_path)
        metadata = {
            "card_name": card_name,
            "global_prompt": global_prompt,
            "deck_seed": deck_seed,
            "card_seed": card_seed,
            "token_sequence": spec.to_token_sequence(),
            "spec": asdict(spec),
            "final_prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "lora_scale": lora_scale,
            "control_image_path": os.path.join(self.layout_dir, f"{base_name}.png"),
            "image_path": image_path,
            "pipeline_branch": branch,
            "symbol_library_dir": self.symbol_generator.library_root,
            "spec_source": type(self.parser).__name__,
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
            results.append(self.generate_card(
                global_prompt=global_prompt,
                card_name=card_name,
                out_dir=out_dir,
                deck_seed=deck_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                width=width,
                height=height,
            ))
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

    def _style_scale_from_z_deck(self, z_deck: torch.Tensor) -> float:
        values = z_deck.detach().float().cpu().flatten()
        if values.numel() == 0:
            return float(self.base_lora_scale)
        signal = torch.tanh(values[:16].mean()).item()
        scale = self.base_lora_scale + 0.12 * signal
        return float(max(0.65, min(1.05, scale)))


def parse_args():
    parser = argparse.ArgumentParser(description="Tarot deck generation pipeline with ControlNet + copy/place for minor arcana.")
    parser.add_argument("--prompt", type=str, required=True, help="Global tarot deck style prompt.")
    parser.add_argument("--out_dir", type=str, default="outputs/tarot_copyplace", help="Directory to save generated images and metadata.")
    parser.add_argument("--card", type=str, default=None, help="Generate only one card, e.g. 'The Fool'. If omitted, generate the full deck.")
    parser.add_argument("--deck_seed", type=int, default=1234, help="Shared deck seed for reproducible generation.")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.2, help="How strongly ControlNet follows the layout image.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--height", type=int, default=768, help="Image height.")
    parser.add_argument("--style_encoder_model", type=str, default="openai/clip-vit-large-patch14", help="Hugging Face model name for style encoding.")
    parser.add_argument("--diffusion_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base diffusion model.")
    parser.add_argument("--controlnet_path", type=str, default="lllyasviel/sd-controlnet-scribble", help="Path to the trained ControlNet checkpoint directory.")
    parser.add_argument("--layout_dir", type=str, default="data/layouts", help="Directory containing control layout images.")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"], help="Force device. By default, auto-selects cuda if available.")
    parser.add_argument("--render_mode", type=str, default="illustration", choices=["illustration", "pixel"], help="Prompt rendering style.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA path for deck-wide style consistency.")
    parser.add_argument("--base_lora_scale", type=float, default=0.85, help="Base LoRA scale; final scale is lightly modulated by z_deck.")
    parser.add_argument("--style_preset", type=str, default="impressionist", choices=["impressionist", "default", "gothic", "art_nouveau", "watercolor"], help="Compact prompt preset.")
    parser.add_argument("--symbol_cache_dir", type=str, default="outputs/symbol_cache", help="Cache directory for raw generated single-symbol images.")
    parser.add_argument("--symbol_library_dir", type=str, default="outputs/symbol_library", help="Library directory for reusable selected symbol prototypes.")
    parser.add_argument("--warmup_symbols_per_suit", type=int, default=4, help="How many symbol prototypes to accumulate per suit before switching to library-first selection.")
    parser.add_argument("--symbol_candidates", type=int, default=4, help="How many symbol candidates to generate during library warmup rounds.")
    parser.add_argument("--store_top_k", type=int, default=3)
    parser.add_argument("--final_select_top_k", type=int, default=3)
    parser.add_argument("--spec_source", type=str, default="rule", choices=["rule", "vllm"], help="How to generate card semantic specs.")
    parser.add_argument("--runpod_api_key", type=str, default=None, help="RunPod API key for vLLM-based spec generation.")
    parser.add_argument("--runpod_endpoint_id", type=str, default=None, help="RunPod endpoint ID for vLLM-based spec generation.")
    parser.add_argument("--vllm_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name metadata for remote vLLM spec generator.")
    parser.add_argument("--vllm_timeout_sec", type=int, default=180, help="Timeout while polling RunPod vLLM jobs.")
    parser.add_argument("--vllm_max_tokens", type=int, default=256, help="Max output tokens for vLLM spec generation.")
    parser.add_argument("--vllm_temperature", type=float, default=0.2, help="Sampling temperature for vLLM spec generation.")
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
        lora_path=args.lora_path,
        base_lora_scale=args.base_lora_scale,
        style_preset=args.style_preset,
        symbol_cache_dir=args.symbol_cache_dir,
        symbol_library_dir=args.symbol_library_dir,
        warmup_symbols_per_suit=args.warmup_symbols_per_suit,
        symbol_candidates=args.symbol_candidates,
        store_top_k=args.store_top_k,
        final_select_top_k=args.final_select_top_k,
        spec_source=args.spec_source,
        runpod_api_key=args.runpod_api_key,
        runpod_endpoint_id=args.runpod_endpoint_id,
        vllm_model_name=args.vllm_model_name,
        vllm_timeout_sec=args.vllm_timeout_sec,
        vllm_max_tokens=args.vllm_max_tokens,
        vllm_temperature=args.vllm_temperature,
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
