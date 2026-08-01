"""
preprocessing.py
=================
Dataset loading + full preprocessing pipeline:
resizing, normalization, dedup, label encoding, augmentation, and the
train/validation/test split. Expects the ImageFolder-style layout produced
by organize_dataset.py (dataset/<class_name>/*.jpg).
"""

import os
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd  
import cv2
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import IMG_SIZE, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED, ORGANIZED_DATASET_DIR


# --------------------------------------------------------------------------
# 1. DATASET LOADING / SUMMARY  (feeds the "Dataset Module" screen)
# --------------------------------------------------------------------------

def build_dataset_index(dataset_dir: str = ORGANIZED_DATASET_DIR) -> pd.DataFrame:
    """
    Walks dataset_dir/<class>/*.jpg and returns a DataFrame with columns:
    filepath, class_name, width, height, channels, file_size_kb, phash
    (phash is a cheap perceptual hash used later for duplicate detection).
    """
    _COLUMNS = ["filepath", "class_name", "width", "height", "channels",
                "file_size_kb", "phash"]

    if not os.path.isdir(dataset_dir):
        print(f"[build_dataset_index] '{dataset_dir}' does not exist — "
              f"returning an empty (but correctly-shaped) DataFrame.")
        return pd.DataFrame(columns=_COLUMNS)

    rows = []
    skipped_unreadable = 0
    for class_name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                continue
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                # cv2 can silently fail to decode (e.g. some WEBP/CMYK JPEGs
                # on Windows OpenCV builds) — skip but keep counting so the
                # cause is visible in the console instead of a silent empty df.
                skipped_unreadable += 1
                continue
            h, w, c = img.shape
            rows.append({
                "filepath": fpath,
                "class_name": class_name,
                "width": w,
                "height": h,
                "channels": c,
                "file_size_kb": round(os.path.getsize(fpath) / 1024, 1),
                "phash": _simple_hash(img),
            })

    if skipped_unreadable:
        print(f"[build_dataset_index] Skipped {skipped_unreadable} file(s) "
              f"OpenCV couldn't decode.")
    if not rows:
        print(f"[build_dataset_index] No readable images found under "
              f"'{dataset_dir}/<class>/'. Check that organize_dataset.py "
              f"actually populated class folders, and that filenames end in "
              f".jpg/.jpeg/.png/.webp/.bmp.")

    # Explicit columns so an empty result still has 'class_name' etc. instead
    # of a zero-column DataFrame that raises KeyError downstream.
    return pd.DataFrame(rows, columns=_COLUMNS)


def _simple_hash(img: np.ndarray) -> str:
    """8x8 average-hash — good enough to flag near-identical duplicates
    without pulling in an extra dependency."""
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (8, 8))
    avg = small.mean()
    bits = (small > avg).flatten()
    return hashlib.md5(bits.tobytes()).hexdigest()


def dataset_summary(df: pd.DataFrame) -> Dict:
    """All the stats the 'Dataset Module' screen needs in one call.
    Safe to call on an empty DataFrame (e.g. dataset/ not populated yet)."""
    if df.empty or "class_name" not in df.columns:
        return {
            "shape": df.shape,
            "num_images": 0,
            "num_classes": 0,
            "class_names": [],
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": {},
            "duplicate_rows": 0,
            "class_counts": {},
            "avg_width": 0,
            "avg_height": 0,
            "avg_file_size_kb": 0,
            "head": df.head(5),
            "is_empty": True,
        }

    return {
        "shape": df.shape,
        "num_images": len(df),
        "num_classes": df["class_name"].nunique(),
        "class_names": sorted(df["class_name"].unique().tolist()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated(subset="phash").sum()),
        "class_counts": df["class_name"].value_counts().to_dict(),
        "avg_width": round(df["width"].mean(), 1) if len(df) else 0,
        "avg_height": round(df["height"].mean(), 1) if len(df) else 0,
        "avg_file_size_kb": round(df["file_size_kb"].mean(), 1) if len(df) else 0,
        "head": df.head(5),
        "is_empty": False,
    }


# --------------------------------------------------------------------------
# 2. CLEANING
# --------------------------------------------------------------------------

def remove_duplicate_images(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows whose perceptual hash has already been seen."""
    before = len(df)
    cleaned = df.drop_duplicates(subset="phash", keep="first").reset_index(drop=True)
    print(f"Removed {before - len(cleaned)} duplicate images "
          f"({before} -> {len(cleaned)}).")
    return cleaned


# --------------------------------------------------------------------------
# 3. LOAD PIXELS + RESIZE + NORMALIZE + ENCODE LABELS
# --------------------------------------------------------------------------

def load_images_and_labels(df: pd.DataFrame, img_size=IMG_SIZE
                            ) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Reads every file in df, resizes to img_size, normalizes to [0,1],
    and label-encodes the class column."""
    X, y_raw = [], []
    for _, row in df.iterrows():
        img = cv2.imread(row["filepath"])
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img.astype("float32") / 255.0  # normalization
        X.append(img)
        y_raw.append(row["class_name"])

    X = np.array(X, dtype="float32")
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y_raw)
    return X, y_int, encoder


def encode_labels_categorical(y_int: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encodes integer labels for categorical-crossentropy training."""
    return to_categorical(y_int, num_classes=num_classes)


# --------------------------------------------------------------------------
# 4. TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------------------------------

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
):
    """
    Splits dataset into Train / Validation / Test.

    Automatically disables stratification if any class has fewer
    than 2 samples.
    """

    labels = y.argmax(axis=1) if y.ndim > 1 else y

    # Count samples in each class
    class_counts = Counter(labels)

    # Check if every class has at least 2 samples
    if min(class_counts.values()) < 2:
        print("⚠ Warning: Some classes have fewer than 2 samples.")
        print("⚠ Using non-stratified split.")
        stratify_labels = None
    else:
        stratify_labels = labels

    # First split (Train / Temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(val_split + test_split),
        random_state=seed,
        stratify=stratify_labels,
    )

    relative_test = test_split / (val_split + test_split)

    temp_labels = y_temp.argmax(axis=1) if y_temp.ndim > 1 else y_temp
    temp_counts = Counter(temp_labels)

    # Check again before second split
    if min(temp_counts.values()) < 2:
        stratify_temp = None
    else:
        stratify_temp = temp_labels

    # Second split (Validation / Test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        random_state=seed,
        stratify=stratify_temp,
    )

    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test
# --------------------------------------------------------------------------
# 5. AUGMENTATION
# --------------------------------------------------------------------------

def build_augmentor() -> ImageDataGenerator:
    """Standard light-to-moderate augmentation suitable for clothing photos
    (avoids heavy color jitter since color is a meaningful feature here)."""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.85, 1.15),
        fill_mode="nearest",
    )


def get_train_generator(X_train, y_train, batch_size: int):
    augmentor = build_augmentor()
    return augmentor.flow(X_train, y_train, batch_size=batch_size, seed=RANDOM_SEED)


if __name__ == "__main__":
    df = build_dataset_index()
    print(dataset_summary(df))
