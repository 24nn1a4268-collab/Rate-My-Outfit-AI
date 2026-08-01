"""
train.py
=========

Training pipeline for the "Rate My Outfit - AI Fashion Critic" project.

This script performs end-to-end training of an outfit classification model
using transfer learning on top of MobileNetV2 (pretrained on ImageNet).

Pipeline stages:
    1. Build a dataset index from the raw dataset directory.
    2. Remove duplicate images to avoid data leakage / bias.
    3. Load images and their corresponding labels into memory.
    4. Encode labels into categorical (one-hot) format.
    5. Split the dataset into train/validation/test partitions.
    6. Build an augmented training data generator.
    7. Build a MobileNetV2-based transfer learning model with a frozen
       backbone and a custom classification head.
    8. Train the classification head (backbone frozen).
    9. Fine-tune the last 30 layers of the backbone.
    10. Evaluate the final model (accuracy, precision, recall, F1,
        classification report, confusion matrix, ROC/AUC).
    11. Persist all artifacts (model, label map, metrics, plots).

Run directly with:
    python train.py
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# Scikit-learn imports for evaluation
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# Plotting
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend suitable for headless training
import matplotlib.pyplot as plt  # noqa: E402

# Local project imports
from processing import (
    build_dataset_index,
    remove_duplicate_images,
    load_images_and_labels,
    encode_labels_categorical,
    split_dataset,
    get_train_generator,
)
from utils import (
    IMG_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    MODEL_PATH,
    LABEL_MAP_PATH,
    REPORTS_DIR,
)


# --------------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------------- #

INITIAL_EPOCHS: int = 40
FINE_TUNE_EPOCHS: int = 20
FINE_TUNE_LAYERS: int = 30
LEARNING_RATE: float = 1e-4
FINE_TUNE_LEARNING_RATE: float = 1e-5
DENSE_UNITS: int = 256 k
DROPOUT_RATE: float = 0.3

# Ensure deterministic behaviour where possible
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# --------------------------------------------------------------------------- #
# Model building
# --------------------------------------------------------------------------- #

def build_model(num_classes: int, img_size: Tuple[int, int] = IMG_SIZE) -> tf.keras.Model:
    """Build a MobileNetV2-based transfer learning model.

    The MobileNetV2 backbone is loaded with ImageNet pretrained weights and
    frozen. A custom classification head is attached on top:
        GlobalAveragePooling2D -> Dense(256, relu) -> Dropout(0.3) ->
        Dense(num_classes, softmax)

    Args:
        num_classes: Number of output classes for the softmax layer.
        img_size: Tuple of (height, width) for the input images.

    Returns:
        A compiled `tf.keras.Model` ready for initial (head-only) training.

    Raises:
        ValueError: If `num_classes` is not a positive integer.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")

    input_shape = (img_size[0], img_size[1], 3)

    # Load MobileNetV2 backbone with ImageNet pretrained weights, no top layer.
    backbone = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # Freeze the backbone so only the classification head trains initially.
    backbone.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense_256")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_0_3")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="outfit_classifier")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def fine_tune(model: tf.keras.Model, num_layers: int = FINE_TUNE_LAYERS) -> tf.keras.Model:
    """Unfreeze the last `num_layers` layers of the MobileNetV2 backbone.

    The backbone is located as the first layer of the functional model that
    has nested layers (i.e. the MobileNetV2 submodel). The last `num_layers`
    layers of that submodel are set to trainable, and the model is
    recompiled with a lower learning rate suitable for fine-tuning.

    Args:
        model: The previously trained `tf.keras.Model` returned by
            `build_model` (and trained via `train_model`).
        num_layers: Number of trailing backbone layers to unfreeze.

    Returns:
        The recompiled `tf.keras.Model`, ready for fine-tuning.

    Raises:
        ValueError: If the MobileNetV2 backbone cannot be located inside
            the given model.
    """
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        raise ValueError("Could not locate MobileNetV2 backbone inside the model.")

    # Unfreeze the whole backbone first, then re-freeze everything except
    # the last `num_layers` layers.
    backbone.trainable = True
    freeze_until = max(len(backbone.layers) - num_layers, 0)
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
    for layer in backbone.layers[freeze_until:]:
        layer.trainable = True

    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _build_callbacks(checkpoint_path: str) -> List[Callback]:
    """Construct the standard set of training callbacks.

    Args:
        checkpoint_path: Filesystem path where the best model checkpoint
            should be saved.

    Returns:
        A list of Keras callbacks: EarlyStopping, ReduceLROnPlateau, and
        ModelCheckpoint.
    """
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    return [early_stopping, reduce_lr, checkpoint]


def train_model(
    model: tf.keras.Model,
    train_generator: Any,
    validation_data: Tuple[np.ndarray, np.ndarray],
    epochs: int,
    checkpoint_path: str,
) -> tf.keras.callbacks.History:
    """Train (or fine-tune) the given model.

    Args:
        model: A compiled `tf.keras.Model`.
        train_generator: A Keras data generator (e.g. from
            `ImageDataGenerator.flow`) yielding batches of
            (images, labels).
        validation_data: Tuple of (X_val, y_val) numpy arrays.
        epochs: Number of epochs to train for.
        checkpoint_path: Path to save the best model checkpoint to.

    Returns:
        The `History` object produced by `model.fit`.

    Raises:
        RuntimeError: If training fails for any reason.
    """
    try:
        callbacks = _build_callbacks(checkpoint_path)
        history = model.fit(
            train_generator,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        return history
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Model training failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Plotting utilities
# --------------------------------------------------------------------------- #


def plot_training_curves(
    histories: List[tf.keras.callbacks.History],
    reports_dir: str = REPORTS_DIR,
) -> None:
    """Plot and save accuracy and loss curves across training phases.

    Multiple `History` objects (e.g. initial training + fine-tuning) are
    concatenated so the curves show the full training trajectory.

    Args:
        histories: List of Keras `History` objects, in chronological order.
        reports_dir: Directory in which to save the resulting plot images.

    Raises:
        ValueError: If `histories` is empty.
    """
    if not histories:
        raise ValueError("histories list must not be empty.")

    os.makedirs(reports_dir, exist_ok=True)

    acc: List[float] = []
    val_acc: List[float] = []
    loss: List[float] = []
    val_loss: List[float] = []

    for history in histories:
        acc.extend(history.history.get("accuracy", []))
        val_acc.extend(history.history.get("val_accuracy", []))
        loss.extend(history.history.get("loss", []))
        val_loss.extend(history.history.get("val_loss", []))

    epochs_range = range(1, len(acc) + 1)

    # Accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.title("Model Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "accuracy_curve.png"))
    plt.close()

    # Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title("Model Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "loss_curve.png"))
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    reports_dir: str = REPORTS_DIR,
) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth integer class labels.
        y_pred: Predicted integer class labels.
        class_names: List of human-readable class names, ordered by index.
        reports_dir: Directory in which to save the resulting plot image.
    """
    os.makedirs(reports_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "confusion_matrix.png"))
    plt.close()


def plot_roc_curves(
    y_true_onehot: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    reports_dir: str = REPORTS_DIR,
) -> Dict[str, float]:
    """Plot and save per-class ROC curves with AUC scores.

    Args:
        y_true_onehot: One-hot encoded ground-truth labels, shape
            (n_samples, n_classes).
        y_pred_proba: Predicted class probabilities, shape
            (n_samples, n_classes).
        class_names: List of human-readable class names, ordered by index.
        reports_dir: Directory in which to save the resulting plot image.

    Returns:
        Dictionary mapping class name to its AUC score.
    """
    os.makedirs(reports_dir, exist_ok=True)

    n_classes = len(class_names)
    auc_scores: Dict[str, float] = {}

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[class_names[i]] = float(roc_auc)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")
        except Exception:  # noqa: BLE001
            # Skip classes for which ROC cannot be computed (e.g. no positives)
            auc_scores[class_names[i]] = float("nan")
            continue

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "roc_curves.png"))
    plt.close()

    return auc_scores


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test_onehot: np.ndarray,
    class_names: List[str],
    reports_dir: str = REPORTS_DIR,
) -> Dict[str, Any]:
    """Evaluate the trained model on the held-out test set.

    Computes accuracy, precision, recall, F1 score, a full classification
    report, confusion matrix, and ROC/AUC curves. Plots are saved as a
    side effect via `plot_confusion_matrix` and `plot_roc_curves`.

    Args:
        model: The trained `tf.keras.Model`.
        X_test: Test set images.
        y_test_onehot: One-hot encoded test set labels.
        class_names: List of human-readable class names, ordered by index.
        reports_dir: Directory in which to save evaluation plots.

    Returns:
        A dictionary containing all computed metrics and the raw
        classification report (as a dict).

    Raises:
        RuntimeError: If evaluation fails for any reason.
    """
    try:
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test_onehot, axis=1)

        accuracy = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        class_report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        plot_confusion_matrix(y_true, y_pred, class_names, reports_dir)
        auc_scores = plot_roc_curves(y_test_onehot, y_pred_proba, class_names, reports_dir)

        metrics: Dict[str, Any] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_scores": auc_scores,
        }

        return {
            "metrics": metrics,
            "classification_report": class_report,
        }
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Model evaluation failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_artifacts(
    model,
    class_names,
    model_path=MODEL_PATH,
    label_map_path=LABEL_MAP_PATH,
):
    """
    Save trained model and label map.
    """

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(label_map_path) or ".", exist_ok=True)

    # Save model
    model.save(model_path)

    # Save label map as:
    # {"0":"dress","1":"hoodie",...}
    label_map = {
        str(i): class_name
        for i, class_name in enumerate(class_names)
    }

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)

    print("Model saved:", model_path)
    print("Label map saved:", label_map_path)

def save_metrics(
    metrics: Dict[str, Any],
    classification_report_dict: Dict[str, Any],
    reports_dir: str = REPORTS_DIR,
) -> None:
    """Save computed metrics and the classification report to disk as JSON.

    Args:
        metrics: Dictionary of scalar/aggregate metrics.
        classification_report_dict: The sklearn classification report as
            a dictionary.
        reports_dir: Directory in which to save the JSON files.

    Raises:
        OSError: If the metrics cannot be written to disk.
    """
    try:
        os.makedirs(reports_dir, exist_ok=True)

        metrics_path = os.path.join(reports_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        report_path = os.path.join(reports_dir, "classification_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(classification_report_dict, f, indent=4)
    except OSError as exc:
        raise OSError(f"Failed to save metrics: {exc}") from exc


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main() -> None:
    """Run the complete end-to-end training pipeline.

    Steps:
        1. Build dataset index and remove duplicate images.
        2. Load images and labels, then encode labels categorically.
        3. Split into train/validation/test sets.
        4. Build an augmented training generator.
        5. Build and train the classification head (frozen backbone).
        6. Fine-tune the last 30 backbone layers.
        7. Evaluate the final model on the test set.
        8. Save plots, metrics, classification report, model, and label map.
        9. Print a final summary to the console.
    """
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

        # ------------------------------------------------------------- #
        # 1. Dataset indexing and cleaning
        # ------------------------------------------------------------- #
        print("[1/9] Building dataset index...")
        dataset_index = build_dataset_index()

        print("[2/9] Removing duplicate images...")
        dataset_index = remove_duplicate_images(dataset_index)

        # ------------------------------------------------------------- #
        # 2. Load images/labels and encode
        # ------------------------------------------------------------- #
        print("[3/9] Loading images and labels...")
        images, labels,label_encoder = load_images_and_labels(dataset_index)

        print("[4/9] Encoding labels categorically...")
        class_names = list(label_encoder.classes_)
        num_classes = len(class_names)

        

        # 3. Pass both required arguments to the encoding function
        labels_onehot = encode_labels_categorical(labels, num_classes)

        # -------------------------------------------------------------#
        # 3. Split dataset
        # ------------------------------------------------------------- #
        print("[5/9] Splitting dataset into train/validation/test...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            images, labels_onehot
        )

        # ------------------------------------------------------------- #
        # 4. Build training generator
        # ------------------------------------------------------------- #
        print("[6/9] Creating augmented training generator...")
        train_generator = get_train_generator(X_train, y_train, batch_size=BATCH_SIZE)

        # ------------------------------------------------------------- #
        # 5. Build and train the model (frozen backbone)
        # ------------------------------------------------------------- #
        print("[7/9] Building model and starting initial training phase...")
        model = build_model(num_classes=num_classes, img_size=IMG_SIZE)

        initial_checkpoint_path = os.path.join(
            os.path.dirname(MODEL_PATH) or ".", "outfit_classifier_initial.h5"
        )
        history_initial = train_model(
            model=model,
            train_generator=train_generator,
            validation_data=(X_val, y_val),
            epochs=INITIAL_EPOCHS,
            checkpoint_path=initial_checkpoint_path,
        )

        # ------------------------------------------------------------- #
        # 6. Fine-tune
        # ------------------------------------------------------------- #
        print("[8/9] Fine-tuning last layers of the backbone...")
        model = fine_tune(model, num_layers=FINE_TUNE_LAYERS)

        fine_tune_checkpoint_path = MODEL_PATH
        history_fine_tune = train_model(
            model=model,
            train_generator=train_generator,
            validation_data=(X_val, y_val),
            epochs=FINE_TUNE_EPOCHS,
            checkpoint_path=fine_tune_checkpoint_path,
        )

        # ------------------------------------------------------------- #
        # 7. Evaluate
        # ------------------------------------------------------------- #
        print("[9/9] Evaluating model on the held-out test set...")
        eval_results = evaluate_model(
            model=model,
            X_test=X_test,
            y_test_onehot=y_test,
            class_names=class_names,
            reports_dir=REPORTS_DIR,
        )
        metrics = eval_results["metrics"]
        classification_report_dict = eval_results["classification_report"]

        # ------------------------------------------------------------- #
        # 8. Save plots, metrics, and artifacts
        # ------------------------------------------------------------- #
        plot_training_curves([history_initial, history_fine_tune], reports_dir=REPORTS_DIR)
        save_metrics(metrics, classification_report_dict, reports_dir=REPORTS_DIR)
        save_artifacts(
            model=model,
            class_names=class_names,
            model_path=MODEL_PATH,
            label_map_path=LABEL_MAP_PATH,
        )

        # ------------------------------------------------------------- #
        # 9. Final summary
        # ------------------------------------------------------------- #
        print("\n===== Training Complete =====")
        print(f"Final Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision      : {metrics['precision']:.4f}")
        print(f"Recall         : {metrics['recall']:.4f}")
        print(f"F1 Score       : {metrics['f1_score']:.4f}")
        print("\nSaved artifacts:")
        print(f"  Model              -> {MODEL_PATH}")
        print(f"  Label map          -> {LABEL_MAP_PATH}")
        print(f"  Metrics            -> {os.path.join(REPORTS_DIR, 'metrics.json')}")
        print(
            "  Classification rpt -> "
            f"{os.path.join(REPORTS_DIR, 'classification_report.json')}"
        )
        print(f"  Accuracy curve     -> {os.path.join(REPORTS_DIR, 'accuracy_curve.png')}")
        print(f"  Loss curve         -> {os.path.join(REPORTS_DIR, 'loss_curve.png')}")
        print(
            "  Confusion matrix   -> "
            f"{os.path.join(REPORTS_DIR, 'confusion_matrix.png')}"
        )
        print(f"  ROC curves         -> {os.path.join(REPORTS_DIR, 'roc_curves.png')}")

    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] Training pipeline failed: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()