"""
Fine-Tuning Script — Transfer Learning with VGG19
Trains a classification head on top of frozen VGG19 layers,
then unfreezes the top conv block for domain adaptation.

Usage:
    python train.py --images_dir ./images --styles_csv ./styles.csv \
                    --epochs 10 --output_model models/fine_tuned_vgg19.keras
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

# NOTE: TensorBoard intentionally removed — it triggers TBNotInstalledError
# at runtime even when the keras callback imports without error.
# Install `tensorboard` separately and add it back if you need TB logs.

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# Data pipeline
def load_metadata(styles_csv: str, images_dir: str) -> pd.DataFrame:
    df = pd.read_csv(styles_csv, on_bad_lines="skip")
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id", "articleType"])
    df["id"] = df["id"].astype(int)

    img_dir = Path(images_dir)
    existing_ids = {int(p.stem) for p in img_dir.glob("*.jpg")}
    df = df[df["id"].isin(existing_ids)].reset_index(drop=True)
    print(f"{len(df)} products with matching images")
    return df


def make_dataset(df: pd.DataFrame, images_dir: str, label_col: str,
                 batch_size: int, training: bool):
    """Build a tf.data pipeline for efficient loading and augmentation."""
    img_dir = Path(images_dir)
    paths = [str(img_dir / f"{row.id}.jpg") for _, row in df.iterrows()]
    labels = df[label_col].values

    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    n_classes = len(le.classes_)

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(encoded)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def load_and_preprocess(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_input(img)
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.85, 1.15)
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=2000, seed=SEED)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, le, n_classes


# Model builder
def build_model(n_classes: int, dropout: float = 0.5):
    base = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False  # frozen during phase 1

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout * 0.6)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    return model, base


# Training loop avec deux phases: head training + fine-tuning (transfer learning)
def train(args):
    # verification directory exists
    output_model_path = Path(args.output_model)
    output_dir = output_model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    phase1_checkpoint = str(output_dir / "checkpoint_phase1.keras")

    # 1. charger metadata
    df = load_metadata(args.styles_csv, args.images_dir)

    counts = df["articleType"].value_counts()
    valid_classes = counts[counts >= 4].index
    df = df[df["articleType"].isin(valid_classes)].reset_index(drop=True)

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df["articleType"]
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # 2. build tf.data datasets
    train_ds, label_encoder, n_classes = make_dataset(
        train_df, args.images_dir, "articleType", BATCH_SIZE, training=True
    )
    val_ds, _, _ = make_dataset(
        val_df, args.images_dir, "articleType", BATCH_SIZE, training=False
    )
    print(f"{n_classes} article type classes")

    # 3. build et compile model
    model, base_model = build_model(n_classes)
    model.compile(
        optimizer=Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(
        print_fn=lambda x: print(x)
        if any(k in x for k in ("Total", "Trainable", "Non-trainable"))
        else None
    )

    # Phase 1: tete d’entrainement uniquement (VGG19 fully frozen)
    print("\nPhase 1: Training classification head (VGG19 frozen)…")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6, verbose=1),
            ModelCheckpoint(phase1_checkpoint, save_best_only=True, verbose=1),
        ],
    )

    # Phase 2: Fine-tune block5 conv layers
    print("\nPhase 2: Unfreezing block5 conv layers for fine-tuning…")
    for layer in base_model.layers:
        if layer.name.startswith("block5"):
            layer.trainable = True

    model.compile(
        optimizer=Adam(1e-5),   # low LR pour preserver les poids pre entraines
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=[
            EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-7, verbose=1),
            ModelCheckpoint(args.output_model, save_best_only=True, verbose=1),
        ],
    )

    print(f"\nFine-tuned model saved to: {args.output_model}")

    # enregistrer label encoder et the model
    encoder_path = output_dir / "label_encoder.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to: {encoder_path}")


# execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune VGG19 on fashion dataset")
    parser.add_argument("--images_dir", default="../dataset/images")
    parser.add_argument("--styles_csv", default="../dataset/styles.csv")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Phase 1 epochs (head training)")
    parser.add_argument("--fine_tune_epochs", type=int, default=5,
                        help="Phase 2 epochs (block5 fine-tuning)")
    # Default changed to place it inside "models" directory
    parser.add_argument("--output_model", default="../models/fine_tuned_vgg19.keras")
    args = parser.parse_args()

    train(args)
    