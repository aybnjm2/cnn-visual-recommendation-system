# StyleLens — CNN Visual E-Commerce Recommender

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)

A deep-learning powered product recommender built on **VGG19 transfer learning** + **TensorFlow**, wrapped in a polished **Streamlit** UI. Upload any product photo and get 10 visually similar recommendations from a 44k-image fashion catalogue.

---

## Architecture Overview

```
Query Image (1080×1440 JPEG)
        │
        ▼
  Preprocessing (resize 224×224, VGG19 mean subtraction)
        │
        ▼
  VGG19 Backbone (ImageNet pretrained)
  ─ Block 1-4: frozen
  ─ Block 5:   fine-tuned on fashion data  ← Transfer Learning
        │
        ▼
  GlobalAvgPool → Dense(512) → Dropout → Dense(256)  [fine-tune head]
  ─ OR ─
  fc1 → fc2  (4096-dim)  [base VGG19 mode]
        │
        ▼
  L2-Normalised Feature Vector
        │
        ▼
  Cosine Similarity vs 44k-product Feature Matrix
        │
        ▼
  Hybrid Score = 0.75 × Visual + 0.25 × Metadata (articleType, colour, gender)
        │
        ▼
  Top-10 Recommendations + Product Cards
```

---

## Project Structure

```
cnn_recommender/
├── app.py                 # Streamlit UI (main entry point)
├── feature_extractor.py   # VGG19 preprocessing & feature extraction
├── recommender.py         # Cosine similarity engine + metadata re-ranking
├── train.py               # Transfer learning fine-tuning script
├── build_index.py         # Pre-compute & cache all product embeddings
├── requirements.txt
└── README.md

dataset/                   # Your data (place alongside cnn_recommender/)
├── images/                # 44441 JPEG files (1163.jpg … 60000.jpg)
└── styles.csv             # id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
# GPU support (optional but recommended for 44k images):
pip install tensorflow[and-cuda]
```

### 2. (Optional) Fine-tune VGG19 on the fashion dataset
This step adapts the model to recognize fashion-specific categories.
```bash
python train.py \
  --images_dir ./images \
  --styles_csv ./styles.csv \
  --epochs 10 \
  --fine_tune_epochs 5 \
  --output_model ./fine_tuned_vgg19.keras
```
**Tip:** Skip this step to use base ImageNet VGG19 — it already gives strong results for colour/texture matching. Fine-tuning improves category-level similarity.

### 3. Build the feature index
Extract and cache 4096-dim embeddings for all 44k products. **This runs once and takes ~30-90 min on CPU, ~5-15 min on GPU.**
```bash
# Base VGG19:
python build_index.py --images_dir ./images --output features.pkl

# Fine-tuned model:
python build_index.py \
  --images_dir ./images \
  --model_path ./fine_tuned_vgg19.keras \
  --output features.pkl
```
The script supports **resuming** — safe to interrupt and re-run.

### 4. Launch the Streamlit app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How It Works

### Image Preprocessing
VGG19 expects **224×224 RGB** inputs. Your 1080×1440 images are:
1. Resized to 224×224 (PIL `LANCZOS` resampling — no padding, aspect ratio adjustable)
2. Converted to `float32` NumPy array
3. Processed with `vgg19.preprocess_input()` — subtracts ImageNet channel means `[103.939, 116.779, 123.68]` in BGR order

### Transfer Learning Strategy (2-Phase)
| Phase | Layers Trained | Learning Rate | Purpose |
|-------|---------------|---------------|---------|
| 1 | New head only (Dense 512→256→N) | 1e-3 | Learn fashion-specific decision boundary |
| 2 | block5_conv1/2/3 + head | 1e-5 | Adapt low-level texture detectors to fashion textures |

### Similarity Search
- Feature vectors are **L2-normalised** at index time
- Similarity = **dot product** (equivalent to cosine similarity for unit vectors)
- **O(N)** scan over the full feature matrix using NumPy broadcasting — fast for 44k products
- For >500k products, swap in FAISS for approximate nearest-neighbour search

### Hybrid Ranking
```
final_score = α × visual_cosine_score + (1-α) × metadata_overlap_score
```
- `α` (default 0.75) is adjustable in the sidebar
- Metadata score weights: articleType (40%), masterCategory (30%), baseColour (20%), gender (10%)

---

## Configuration (Sidebar)

| Setting | Default | Description |
|---------|---------|-------------|
| Images directory | `./images` | Path to your image folder |
| Features file | `./features.pkl` | Cached embeddings (built by `build_index.py`) |
| Styles CSV | `./styles.csv` | Product metadata |
| Fine-tuned model | _(empty)_ | Optional path to `.keras` model |
| Visual weight α | 0.75 | Balance visual vs. metadata similarity |
| Top-K | 10 | Number of recommendations |

---

## Performance Notes

- **Feature extraction**: ~150ms/image on CPU, ~20ms on GPU
- **Index build** (44k images): ~2h on CPU, ~15min on GPU
- **Inference** (single query): <1s including similarity search over 44k vectors
- **Memory**: ~44k × 4096 × 4 bytes ≈ **700 MB** for the feature matrix

---

## Why This Project Is Useful

| Feature | Benefit |
|---------|---------|
| **Transfer Learning** | Leverages ImageNet pretrained VGG19 — no need to train from scratch |
| **Hybrid Scoring** | Combines visual similarity with metadata for more relevant results |
| **Resumable Indexing** | Build the feature index in chunks — interrupt and resume anytime |
| **Customizable Weights** | Adjust visual vs. metadata balance via sidebar slider |
| **Pre-trained Model Included** | Comes with `fine_tuned_vgg19.keras` for immediate use |

---

## Where Users Can Get Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Documentation**: This README covers setup and configuration; see code comments for API details
- **Dataset**: The fashion product images are from [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

---

## Who Maintains and Contributes

This project was developed as a deep learning demonstration project.

### Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure tests pass and code follows the existing style.

---

## License

This project is available for educational and personal use. See the dataset [NOTICE](dataset/NOTICE ABOUT DATASET.txt) for data licensing terms.
