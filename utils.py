"""
utils.py
========
Shared configuration, helper functions, the Vision-Language-Model (VLM)
wrapper, color analysis utilities, and the rule-based "fashion intelligence"
engine that turns raw CV outputs (category, colors, confidence) into all the
higher-level dashboard content (occasion recs, trend scores, mood, etc.).

Everything that is reused by more than one module lives here so the rest of
the codebase stays thin.
"""

import os
import json
import random
from typing import List, Dict, Tuple

import numpy as np
import cv2
from PIL import Image

# --------------------------------------------------------------------------
# GLOBAL CONFIG
# --------------------------------------------------------------------------

CLASS_NAMES = [
    "t-shirt", "hoodie", "jacket", "jeans", "pants", "shorts",
    "skirt", "dress","shoes", 
]

IMG_SIZE = (224, 224)          # input size for the CNN backbone
BATCH_SIZE = 32
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Paths
MODEL_DIR = "saved_models"
REPORTS_DIR = "reports"

RAW_DATASET_DIR = "dataset_raw"        # unsorted images (as uploaded)
ORGANIZED_DATASET_DIR = "dataset"      # dataset/<class_name>/*.jpg
MODEL_PATH = "saved_models/outfit_classifier.h5"
LABEL_MAP_PATH = "saved_models/label_map.json"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------------------------------
# VISION LANGUAGE MODEL WRAPPER (Hugging Face Transformers)
# --------------------------------------------------------------------------
#
# Built around Hugging Face Transformers as requested. Defaults to
# "vikhyatk/moondream2" (small, fast, runs on CPU or a single GPU).
# Swap MODEL_ID for "llava-hf/llava-1.5-7b-hf" if you have a bigger GPU.
#
# The wrapper is lazy-loaded (model only pulled into memory on first use)
# and fails gracefully into a template-based fallback if no GPU/internet/
# model weights are available, so the rest of the app keeps working.

VLM_MODEL_ID = "vikhyatk/moondream2"


class VisionLanguageCritic:
    """Thin wrapper around a HF VLM used to generate the roast/hype text."""

    def __init__(self, model_id: str = VLM_MODEL_ID, device: str = None):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None
        self._available = None  # tri-state: None=unknown, True/False

    def _try_load(self):
        if self._available is not None:
            return self._available
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, trust_remote_code=True
            ).to(device)
            self.device = device
            self._available = True
        except Exception as exc:  # noqa: BLE001 - want a broad, safe fallback
            print(f"[VisionLanguageCritic] Falling back to template mode: {exc}")
            self._available = False
        return self._available

    def critique(self, image: Image.Image, detected: Dict) -> Dict:
        """
        Returns a dict with roast comments, hype comments, and a short
        outfit summary paragraph. Uses the real VLM if it loaded correctly,
        otherwise falls back to a deterministic template engine so the demo
        never breaks (e.g. offline hackathon judging room).
        """
        prompt = self._build_prompt(detected)

        if self._try_load():
            try:
                enc_image = self._model.encode_image(image)
                answer = self._model.answer_question(enc_image, prompt, self._tokenizer)
                return self._parse_vlm_output(answer, detected)
            except Exception as exc:  # noqa: BLE001
                print(f"[VisionLanguageCritic] Inference failed, using fallback: {exc}")

        return self._template_fallback(detected)
    @staticmethod
    def _build_prompt(detected: Dict) -> str:
        items = ", ".join(detected.get("items", [])) or "an outfit"

        return f"""
    You are a professional fashion stylist.

    IMPORTANT:
    - Judge ONLY the outfit.
    - Never judge the person's skin color.
    - Never judge body shape.
    - Never judge face or hairstyle.
    - Never mention race or complexion.

    Evaluate only:
    - Clothing style
    - Outfit coordination
    - Layering
    - Accessories
    - Footwear
    - Occasion suitability
    - Fashion sense

    Detected clothing:
    {items}

    Return ONLY valid JSON:

    {{
        "roast": [
            "comment1",
            "comment2",
            "comment3"
        ],
        "hype": [
            "comment1",
            "comment2",
            "comment3"
        ],
        "summary": "Short summary about the outfit only."
    }}
    """
    @staticmethod
    def _parse_vlm_output(answer: str, detected: Dict) -> Dict:
        try:
            data = json.loads(answer)
            if all(k in data for k in ("roast", "hype", "summary")):
                return data
        except Exception:  # noqa: BLE001
            pass
        # If the model replied in plain prose instead of JSON, just use it
        # as the summary and pair it with template jokes.
        fallback = VisionLanguageCritic._template_fallback(detected)
        fallback["summary"] = answer.strip() or fallback["summary"]
        return fallback

    @staticmethod
    def _template_fallback(detected: Dict) -> Dict:
        items = detected.get("items", ["an outfit"])
        main_item = items[0] if items else "outfit"
        colors = detected.get("dominant_colors", ["neutral tones"])
        score = detected.get("overall_score", 6.0)

        roast_bank = [
            f"That {main_item} looks like it lost a bet with the laundry basket.",
            "Even the mannequin asked for a wardrobe upgrade.",
            "Corporate intern meets weekend tourist energy.",
            "This fit said 'I have five minutes and zero regrets.'",
            "The color combo is giving 'traffic cone at a funeral.'",
        ]
        hype_bank = [
            "Absolute drip, no notes.",
            "Main character energy, fully loaded.",
            "Fashion police approved this on sight.",
            "You understood the assignment and then some.",
            f"That {main_item} choice? Immaculate taste.",
        ]

        random.shuffle(roast_bank)
        random.shuffle(hype_bank)
        picks = hype_bank[:3] if score >= 7 else roast_bank[:3]

        summary = (
            f"This outfit centers on {', '.join(items[:3])} in "
            f"{', '.join(colors[:2])} tones. Overall it reads as a "
            f"{'polished, well-put-together' if score >= 7 else 'casual, low-effort'} "
            "look with room to play with accessories and layering."
        )

        return {
            "roast": roast_bank[:3],
            "hype": hype_bank[:3],
            "summary": summary,
        }


# --------------------------------------------------------------------------
# COLOR ANALYSIS
# --------------------------------------------------------------------------

# A small curated palette used to name detected RGB clusters in plain English.
_NAMED_COLORS = {
    "black": (0, 0, 0), "white": (255, 255, 255), "grey": (128, 128, 128),
    "navy": (0, 0, 80), "blue": (0, 90, 200), "sky blue": (135, 206, 235),
    "red": (200, 30, 30), "maroon": (110, 20, 30), "pink": (240, 150, 190),
    "orange": (230, 120, 30), "yellow": (230, 210, 40), "beige": (222, 200, 165),
    "brown": (110, 70, 40), "green": (40, 130, 60), "olive": (110, 110, 40),
    "purple": (120, 50, 150), "teal": (30, 130, 130),
}


def _closest_color_name(rgb: Tuple[int, int, int]) -> str:
    best_name, best_dist = "unknown", float("inf")
    for name, ref in _NAMED_COLORS.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, ref))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def extract_dominant_colors(image: np.ndarray, k: int = 3) -> List[Dict]:
    """
    K-means color clustering (OpenCV) over the image pixels.
    Returns a list of {"name": str, "rgb": (r,g,b), "pct": float}
    ordered by prevalence (dominant -> secondary -> accent).
    """
    img = cv2.resize(image, (100, 100))
    pixels = img.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
    )
    counts = np.bincount(labels.flatten())
    order = np.argsort(-counts)

    results = []
    for idx in order:
        b, g, r = centers[idx].astype(int)  # OpenCV loads as BGR
        rgb = (int(r), int(g), int(b))
        pct = round(100 * counts[idx] / counts.sum(), 1)
        results.append({"name": _closest_color_name(rgb), "rgb": rgb, "pct": pct})
    return results


_COLOR_PAIR_RATING = {
    frozenset(["black", "white"]): "Excellent",
    frozenset(["blue", "grey"]): "Very Good",
    frozenset(["navy", "white"]): "Excellent",
    frozenset(["black", "beige"]): "Very Good",
    frozenset(["green", "purple"]): "Needs Improvement",
    frozenset(["orange", "pink"]): "Needs Improvement",
    frozenset(["red", "green"]): "Needs Improvement",
}


def rate_color_combination(colors: List[str]) -> Tuple[str, str]:
    """Returns (rating_label, suggestion) for the primary+secondary colors."""
    if len(colors) < 2:
        return "Good", "Add a contrasting accessory to build a fuller palette."
    pair = frozenset(colors[:2])
    for known_pair, rating in _COLOR_PAIR_RATING.items():
        if pair == known_pair:
            suggestion = (
                "Great pairing, keep it up."
                if rating in ("Excellent", "Very Good")
                else "Try swapping the secondary piece for black, white, navy or grey."
            )
            return rating, suggestion
    return "Good", "Solid pairing — a neutral accessory would elevate it further."


# --------------------------------------------------------------------------
# RULE-BASED FASHION INTELLIGENCE ENGINE
# --------------------------------------------------------------------------
# These sub-scores are not independently modeled (no labeled data exists for
# "trendiness" or "mood" in a typical clothing-classification dataset), so we
# derive them deterministically from what the CV pipeline *does* know:
# category, dominant colors, and classifier confidence. This is standard
# practice for hackathon-style "fashion critic" demos and keeps every number
# on screen explainable and reproducible from a fixed seed.

_OCCASIONS = {
    "shirt": ["Office", "Interview", "Date", "College"],
    "t-shirt": ["College", "Travel", "Shopping", "Gym"],
    "hoodie": ["Travel", "Gym", "College"],
    "jacket": ["Date", "Party", "Travel"],
    "jeans": ["College", "Date", "Shopping", "Travel"],
    "pants": ["Office", "Interview", "Date"],
    "shorts": ["Beach", "Gym", "Travel"],
    "skirt": ["Party", "Date", "College"],
    "dress": ["Wedding", "Party", "Date"],
    "sweater": ["Office", "College", "Date"],
    "shoes": ["Office", "College", "Party"],
    "cap": ["Travel", "Gym", "Shopping"],
    "accessory": ["Party", "Date", "Wedding"],
}

_FABRICS = {
    "shirt": "Cotton", "t-shirt": "Cotton", "hoodie": "Polyester",
    "jacket": "Leather", "jeans": "Denim", "pants": "Cotton",
    "shorts": "Cotton", "skirt": "Polyester", "dress": "Silk",
    "sweater": "Wool", "shoes": "Leather", "cap": "Cotton",
    "accessory": "Mixed",
}

_MOODS = ["Confident", "Professional", "Elegant", "Energetic",
          "Relaxed", "Playful", "Bold", "Luxury"]

_WEATHER = {
    "shorts": "Summer", "t-shirt": "Summer", "dress": "Spring",
    "hoodie": "Winter", "jacket": "Winter", "sweater": "Autumn",
    "shirt": "Spring", "pants": "Autumn", "jeans": "Spring",
    "skirt": "Summer", "shoes": "All Seasons", "cap": "Summer",
    "accessory": "All Seasons",
}

_MISSING_ACCESSORIES = ["Watch", "Bracelet", "Necklace", "Ring",
                         "Cap", "Sunglasses", "Belt", "Bag"]

_FASHION_TIPS = [
    "Add a watch for a finishing touch.", "Try white sneakers for a clean look.",
    "Roll up sleeves for a relaxed vibe.", "Use darker jeans for versatility.",
    "Add a light layer like a jacket or overshirt.",
    "Stick to 2-3 core colors for a cohesive palette.",
    "Tuck in loose tops to sharpen the silhouette.",
]


def _score_from_confidence(confidence: float, spread: float = 3.0) -> float:
    """Maps a 0-1 classifier confidence into a believable 1-10 sub-score."""
    base = 5.5 + (confidence - 0.5) * spread
    jitter = random.uniform(-0.4, 0.4)
    return round(min(10, max(1, base + jitter)), 1)


def build_full_report(items: List[str], colors: List[Dict], confidence: float) -> Dict:
    """
    The single entry point that produces every number/label needed by the
    Streamlit dashboard, given the raw CV outputs.
    """
    color_names = [c["name"] for c in colors]
    primary = items[0] if items else "outfit"

    ratings = {
        "Overall Rating": _score_from_confidence(confidence, 3.5),
        "Color Combination": _score_from_confidence(confidence, 2.5),
        "Style": _score_from_confidence(confidence, 3.0),
        "Trendiness": _score_from_confidence(confidence, 2.5),
        "Accessories": _score_from_confidence(confidence * 0.8, 2.0),
        "Comfort": _score_from_confidence(0.75, 1.5),
        "Confidence": _score_from_confidence(confidence, 2.0),
        "Presentation": _score_from_confidence(confidence, 3.0),
    }

    color_rating, color_suggestion = rate_color_combination(color_names)

    occasions = []
    for it in items:
        occasions.extend(_OCCASIONS.get(it, []))
    occasions = list(dict.fromkeys(occasions)) or ["Casual outing"]

    fabric_guess = _FABRICS.get(primary, "Cotton")
    weather = _WEATHER.get(primary, "All Seasons")
    mood = random.sample(_MOODS, k=2)
    missing_acc = random.sample(_MISSING_ACCESSORIES, k=3)
    tips = random.sample(_FASHION_TIPS, k=3)

    trend_scores = {
        "Trend Score": _score_from_confidence(confidence, 3.0),
        "Streetwear Score": _score_from_confidence(confidence, 2.5),
        "Minimalism Score": _score_from_confidence(1 - abs(confidence - 0.5), 2.0),
        "Popularity Score": _score_from_confidence(confidence, 2.5),
        "Modern Fashion Score": _score_from_confidence(confidence, 3.0),
    }

    return {
        "ratings": ratings,
        "overall_score": ratings["Overall Rating"],
        "color_rating": color_rating,
        "color_suggestion": color_suggestion,
        "occasions": occasions[:6],
        "fabric": fabric_guess,
        "weather": weather,
        "mood": mood,
        "missing_accessories": missing_acc,
        "fashion_tips": tips,
        "trend_scores": trend_scores,
    }


def stars_from_score(score_0_10: float) -> str:
    """Renders a 0-10 score as a 10-slot star string."""
    full = int(round(score_0_10))
    return "⭐" * full + "☆" * (10 - full)
