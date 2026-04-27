"""
Index Builder — Pre-compute VGG19 features for all 44k products.
Usage (with base VGG19):
    python build_index.py --images_dir ./images --output features.pkl
Usage (with fine-tuned model):
    python build_index.py --images_dir ./images \
                          --model_path ./fine_tuned_vgg19.keras \
                          --output features.pkl
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG19
from pathlib import Path

from feature_extractor import (
    build_feature_extractor,
    build_feature_database,
)


def build_extractor_from_fine_tuned(model_path: str) -> Model:
    """
    Load a fine-tuned model and re-export its fc2-equivalent layer
    (the 256-dim dense layer before the softmax head).
    """
    full_model = load_model(model_path)
    full_model.summary()

    # The last Dense before softmax is index -2
    feature_layer = full_model.layers[-3]   # 256-dim dense
    extractor = Model(
        inputs=full_model.input,
        outputs=feature_layer.output,
        name="fine_tuned_extractor",
    )
    extractor.trainable = False
    print(f"Fine-tuned extractor: output shape = {extractor.output_shape}")
    return extractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="../dataset/images")
    parser.add_argument("--model_path", default="../models/fine_tuned_vgg19.keras")
    parser.add_argument("--output", default="../features/features.pkl")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.model_path and Path(args.model_path).exists():
        print(f"Using fine-tuned model: {args.model_path}")
        extractor = build_extractor_from_fine_tuned(args.model_path)
    else:
        print("Using base VGG19 (ImageNet weights)")
        extractor = build_feature_extractor()

    build_feature_database(
        model=extractor,
        images_dir=args.images_dir,
        output_path=args.output,
        batch_size=args.batch_size,
    )
    print("Index built successfully.")
