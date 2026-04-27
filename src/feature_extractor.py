"""
Feature Extractor using VGG19 Transfer Learning
Extracts deep visual features from product images for similarity search.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import pickle
from pathlib import Path
from tqdm import tqdm

# VGG19 entraine sur 224x224 RGB images
IMG_SIZE = (224, 224)
FEATURE_LAYER = "fc2"  # Second fully connected layer — 4096-dim feature vector


def build_feature_extractor():
    """
    Build VGG19-based feature extractor.
    Uses weights pretrained on ImageNet; strips the classification head
    and outputs the fc2 layer (4096-dim embedding).
    """
    base_model = VGG19(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    return Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(FEATURE_LAYER).output,
        name="vgg19_feature_extractor"
    )


def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load and preprocess a single image to VGG19 spec.
    - Resize to 224x224 (center-crop strategy for non-square inputs)
    - Convert to float32 array
    - Apply VGG19-specific mean subtraction (BGR channel order)
    """
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def extract_single_feature(model: Model, img_path: str) -> np.ndarray:
    """Return normalized 4096-dim feature vector for one image."""
    x = preprocess_image(img_path)
    feat = model.predict(x, verbose=0)[0]
    return feat / (np.linalg.norm(feat) + 1e-8)


def build_feature_database(
    model: Model,
    images_dir: str,
    output_path: str = "features.pkl",
    batch_size: int = 32,
):
    """
    Extract and cache features for every image in images_dir.
    Saves a dict  { image_id (int): feature_vector (np.ndarray) }  to disk.

    Supports resuming: already-processed ids are skipped.
    """
    images_dir = Path(images_dir)
    output_path = Path(output_path)
    feature_db = {}

    img_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    for i in tqdm(range(0, len(img_files), batch_size)):
        batch_paths = img_files[i : i + batch_size]
        batch_arrays = [preprocess_image(str(p)) for p in batch_paths]
        
        batch_input = np.vstack(batch_arrays)
        feats = model.predict(batch_input, verbose=0)
        feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)

        for p, feat in zip(batch_paths, feats):
            feature_db[int(p.stem)] = feat

    with open(output_path, "wb") as f:
        pickle.dump(feature_db, f)


def _save(db, path):
    with open(path, "wb") as f:
        pickle.dump(db, f)


def load_feature_database(path: str = "features.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
