import argparse
import os
import shutil

from PIL import Image
from tqdm import tqdm

from utils import CLASS_NAMES, RAW_DATASET_DIR, ORGANIZED_DATASET_DIR

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


def load_classifier():
    """Lazy import so the rest of the project works even without torch/CLIP
    installed until this specific script is run."""
    from transformers import pipeline
    return pipeline("zero-shot-image-classification", model=CLIP_MODEL_ID)


def organize(src_dir: str, dst_dir: str, min_confidence: float = 0.05):
    os.makedirs(dst_dir, exist_ok=True)
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(dst_dir, cls), exist_ok=True)

    classifier = load_classifier()
    candidate_labels = [f"a photo of a {c}" for c in CLASS_NAMES]

    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

    skipped = 0
    for fname in tqdm(files, desc="Auto-labeling images"):
        fpath = os.path.join(src_dir, fname)
        try:
            image = Image.open(fpath).convert("RGB")
        except Exception:
            skipped += 1
            continue

        preds = classifier(image, candidate_labels=candidate_labels)
        top = preds[0]
        if top["score"] < min_confidence:
            skipped += 1
            continue

        label = CLASS_NAMES[candidate_labels.index(top["label"])]
        shutil.copy(fpath, os.path.join(dst_dir, label, fname))

    print(f"Done. {len(files) - skipped}/{len(files)} images sorted into "
          f"'{dst_dir}/<class>/'. {skipped} skipped (low confidence or unreadable).")
    print("Tip: quickly skim each class folder and drag out obvious mistakes "
          "before training — zero-shot labels are a strong starting point, not perfect.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-sort raw images into class folders.")
    parser.add_argument("--src", default=RAW_DATASET_DIR, help="Folder of raw unlabeled images")
    parser.add_argument("--dst", default=ORGANIZED_DATASET_DIR, help="Output ImageFolder-style dataset")
    parser.add_argument("--min-confidence", type=float, default=0.15)
    args = parser.parse_args()
    organize(args.src, args.dst, args.min_confidence)
