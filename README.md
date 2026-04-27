# CNN Visual Recommendation System

A deep learning-based visual similarity search system for fashion products. Upload a product image and get 10 visually similar recommendations using CNN feature extraction (VGG19).

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)

## What the Project Does

This project implements a visual fashion product recommender that:

- **Extracts deep visual features** from product images using VGG19 transfer learning
- **Builds a feature index** for ~44k fashion products
- **Provides similarity search** via cosine similarity on 4096-dim embeddings
- **Offers hybrid scoring** combining visual similarity (70%) with metadata matching (30%)
- **Delivers an interactive Streamlit UI** for image upload and recommendation display

## Why the Project Is Useful

- **No training required** — Use pre-computed ImageNet weights for immediate results
- **Fine-tuning support** — Train on your own product categories for domain adaptation
- **Fast inference** — Pre-computed feature index enables sub-second similarity search
- **Hybrid recommendations** — Combines visual features with metadata (type, color, category)
- **Production-ready** — Clean Python package with proper CLI interfaces

## Quick Start

### 1. Clone and Setup

```bash
# Navigate to project directory
cd cnn-visual-recommendation-system

# Activate virtual environment
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate   # Linux/macOS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv sync
```

### 3. Run the Application

```bash
streamlit run src/app.py
```

The app will open at `http://localhost:8501`. Upload a product image to get 10 similar recommendations.

## Project Structure

```
cnn-visual-recommendation-system/
├── src/
│   ├── app.py              # Streamlit web UI
│   ├── build_index.py      # Build feature index from images
│   ├── feature_extractor.py # VGG19 feature extraction
│   ├── recommender.py      # Similarity search engine
│   └── train.py            # Fine-tuning script
├── dataset/
│   ├── images/             # Product images (44k+)
│   └── styles.csv         # Product metadata
├── features/
│   └── features.pkl       # Pre-computed feature index
├── models/
│   ├── checkpoint_phase1.keras
│   └── fine_tuned_vgg19.keras
├── pyproject.toml
└── requirements.txt
```

## Usage Examples

### Building a Feature Index

```bash
# Using base VGG19 (ImageNet weights)
python src/build_index.py --images_dir ./dataset/images --output ./features/features.pkl

# Using fine-tuned model
python src/build_index.py --images_dir ./dataset/images \
                          --model_path ./models/fine_tuned_vgg19.keras \
                          --output ./features/features.pkl
```

### Fine-Tuning the Model

```bash
python src/train.py --images_dir ./dataset/images \
                    --styles_csv ./dataset/styles.csv \
                    --epochs 10 \
                    --output_model models/fine_tuned_vgg19.keras
```

### Running the Recommender

```python
from src.recommender import ProductRecommender, load_styles
from src.feature_extractor import extract_single_feature, build_feature_extractor
import pickle

# Load data
with open("features/features.pkl", "rb") as f:
    feature_db = pickle.load(f)
styles_df = load_styles("dataset/styles.csv")

# Build recommender
recommender = ProductRecommender(feature_db, styles_df, alpha=0.7)

# Get recommendations
model = build_feature_extractor()
query_vector = extract_single_feature(model, "path/to/uploaded/image.jpg")
results = recommender.recommend(query_vector, top_k=10)
print(results)
```

## Key Features

| Feature | Description |
|---------|-------------|
| **VGG19 Transfer Learning** | Uses pre-trained ImageNet weights for feature extraction |
| **4096-dim Embeddings** | Deep visual representations from fc2 layer |
| **Cosine Similarity** | Efficient similarity metric for visual search |
| **Hybrid Scoring** | Combines visual (α=0.7) + metadata (1-α=0.3) scores |
| **Batch Processing** | Extract features for thousands of images efficiently |
| **Fine-tuning Support** | Domain adaptation with custom classification head |

## Dataset

The project uses the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle:

- **44,000+ product images** (various sizes, stored as JPG/PNG)
- **Metadata CSV** with: id, gender, category, article type, color, season, year, usage

## Requirements

- Python 3.12+
- TensorFlow 2.13+
- Streamlit 1.32+
- NumPy, Pandas, scikit-learn, Pillow, tqdm

See [requirements.txt](requirements.txt) for full dependency list.

## Who Maintains and Contributes

This project was developed as a deep learning demonstration for fashion product recommendation.

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Support

- Open an issue for bug reports or feature requests
- Check existing issues before creating new ones

## License

This project is for educational purposes. See the original dataset's license for data usage terms.
