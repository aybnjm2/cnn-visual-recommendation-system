"""
CNN Visual E-Commerce Recommender — Streamlit UI
Upload a product image get 10 visually similar recommendations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from PIL import Image
import io
import time

# Page config
st.set_page_config(
    page_title="Projet CNN Deep Learning - Recommender System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --cream: #F8F4EE;
        --ink: #1A1510;
        --terracotta: #C4622D;
        --sage: #7A8C6E;
        --gold: #C9A84C;
        --light-gray: #EBEBEB;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--cream);
        color: var(--ink);
    }

    /* Sidebar Lockdown - This makes it static */
    [data-testid="collapsedControl"] {
        display: none; /* Hides the 'X' button and the 'Arrow' button */
    }
    
    section[data-testid="stSidebar"] {
        min-width: 350px !important;
        max-width: 350px !important;
        background-color: #1A1510 !important;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Hero title styling */
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 900;
        color: var(--ink);
        line-height: 1.1;
    }

    .hero-accent { color: var(--terracotta); }

    /* Buttons */
    .stButton > button {
        background-color: var(--terracotta) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100%;
    }

    /* Clean up UI */
    header { visibility: hidden; } /* Header hidden is safe now because we don't want the toggle */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* Product cards and other styles... */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFA500;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid var(--terracotta);
        display: inline-block;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        background: rgba(122, 140, 110, 0.15);
        color: var(--sage);
        border: 1px solid rgba(122, 140, 110, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
    }
    
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.7rem;
        margin-right: 4px;
    }
    .badge-color { background: #F0EDE8; color: #5A4F44; }
    .badge-type { background: rgba(196, 98, 45, 0.1); color: var(--terracotta); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Config
DEFAULT_IMAGES_DIR = "../dataset/images"
DEFAULT_FEATURES_PATH = "../features/features.pkl"
DEFAULT_STYLES_CSV = "../dataset/styles.csv"
DEFAULT_FINE_TUNED_MODEL = "../models/fine_tuned_vgg19.keras"
# Session state init
if "feature_db" not in st.session_state:
    st.session_state.feature_db = None
if "styles_df" not in st.session_state:
    st.session_state.styles_df = None
if "recommender" not in st.session_state:
    st.session_state.recommender = None
if "extractor" not in st.session_state:
    st.session_state.extractor = None


# Loaders
@st.cache_resource(show_spinner=False)
def load_extractor(model_path=None):
    from feature_extractor import build_feature_extractor
    from tensorflow.keras.models import Model, load_model

    if model_path and Path(model_path).exists():
        full_model = load_model(model_path)
        feature_layer = full_model.layers[-3]
        extractor = Model(inputs=full_model.input, outputs=feature_layer.output)
        extractor.trainable = False
    else:
        extractor = build_feature_extractor()
    return extractor


@st.cache_resource(show_spinner=False)
def load_resources(features_path, styles_csv, alpha):
    from recommender import ProductRecommender, load_styles

    feature_db = None
    styles_df = None

    if Path(features_path).exists():
        with open(features_path, "rb") as f:
            feature_db = pickle.load(f)

    if Path(styles_csv).exists():
        styles_df = load_styles(styles_csv)

    recommender = None
    if feature_db and styles_df is not None:
        recommender = ProductRecommender(feature_db, styles_df, alpha=alpha)

    return feature_db, styles_df, recommender


# Sidebar
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    images_dir = st.text_input("Images directory", DEFAULT_IMAGES_DIR)
    features_path = st.text_input("Features file (.pkl)", DEFAULT_FEATURES_PATH)
    styles_csv = st.text_input("Styles CSV path", DEFAULT_STYLES_CSV)
    model_path = st.text_input("Fine-tuned model (optional)", DEFAULT_FINE_TUNED_MODEL)

    st.markdown("---")
    st.markdown("### Ranking Weights")
    alpha = st.slider(
        "Visual similarity weight",
        min_value=0.0, max_value=1.0, value=0.75, step=0.05,
        help="1.0 = pure visual | 0.0 = pure metadata"
    )
    top_k = st.slider("Number of recommendations", 5, 20, 10, step=1)

    st.markdown("---")
    st.markdown("### Build Index")
    st.caption("Run once to extract features for all products.")
    if st.button("Build Feature Index", use_container_width=True):
        st.info("Run `python build_index.py --images_dir ./images` in your terminal.")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem; color:#9C8E80; margin-top:1rem;">'
        "CNN Deep Learning Projet - EMSI<br>VGG19 Transfer Learning<br>TensorFlow + Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# Load resources
with st.spinner("Loading resources…"):
    feature_db, styles_df, recommender = load_resources(features_path, styles_csv, alpha)
    extractor = load_extractor(model_path if model_path else None)


# Hero
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown(
        '<div class="hero-title"><span class="hero-accent">CNN Recommender</span></div>'
        '<div class="hero-sub">Système de recommandation basé sur des CNN pour les stores e-commerce.</div>',
        unsafe_allow_html=True,
    )
with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    if recommender:
        n = len(feature_db) if feature_db else 0
        st.markdown(
            f'<div class="status-pill">{n:,} products indexed</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill" style="color:#C4622D; border-color:rgba(196,98,45,0.3); background:rgba(196,98,45,0.08);">'
            "⚠️ Index not loaded"
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# Upload section
upload_col, preview_col = st.columns([1, 1], gap="large")

with upload_col:
    st.markdown('<div class="section-header">Upload Product Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop any product photo — clothing, shoes, accessories…",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        query_image = Image.open(uploaded_file).convert("RGB")
        st.markdown("<br>", unsafe_allow_html=True)

        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.metric("Width", f"{query_image.width}px")
        with meta_col2:
            st.metric("Height", f"{query_image.height}px")

with preview_col:
    if uploaded_file:
        st.markdown('<div class="section-header">Query Image</div>', unsafe_allow_html=True)
        st.image(query_image, use_container_width=True)


# Recommend button
st.markdown("<br>", unsafe_allow_html=True)

if uploaded_file:
    if not recommender:
        st.warning(
            "Feature index not found. Please run `python build_index.py` first "
            "to pre-compute product embeddings, then restart the app."
        )
    else:
        if st.button(f"Find {top_k} Similar Products", use_container_width=False):
            with st.spinner("Extracting visual features…"):
                from feature_extractor import extract_single_feature
                import tempfile

                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                query_vec = extract_single_feature(extractor, tmp_path)
                os.unlink(tmp_path)

            with st.spinner("Searching 44k products…"):
                results_df = recommender.recommend(query_vec, top_k=top_k)

            st.markdown("---")
            st.markdown(
                f'<div class="section-header">Top {top_k} Recommendations</div>',
                unsafe_allow_html=True,
            )

            # Display as grid (5 per row)
            cols_per_row = 5
            for row_start in range(0, len(results_df), cols_per_row):
                row_items = results_df.iloc[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row, gap="medium")

                for col, (_, item) in zip(cols, row_items.iterrows()):
                    with col:
                        pid = int(item["id"])
                        img_path = Path(images_dir) / f"{pid}.jpg"

                        score = item.get("final_score", item.get("visual_score", 0.0))
                        score_pct = min(int(score * 100), 100)

                        name = item.get("productDisplayName", f"Product #{pid}")
                        article = item.get("articleType", "")
                        colour = item.get("baseColour", "")
                        gender = item.get("gender", "")
                        category = item.get("masterCategory", "")

                        # Card HTML
                        if img_path.exists():
                            pil_img = Image.open(img_path).convert("RGB")
                            img_buf = io.BytesIO()
                            pil_img.thumbnail((300, 400))
                            pil_img.save(img_buf, format="JPEG", quality=85)
                            st.image(img_buf.getvalue(), use_container_width=True)
                        else:
                            st.image(
                                "https://via.placeholder.com/200x267?text=No+Image",
                                use_container_width=True,
                            )

                        st.markdown(
                            f"""
                            <div style="padding: 0.4rem 0;">
                                <div style="font-family:'Playfair Display',serif; font-size:0.82rem;
                                            font-weight:700; color:#FFFFFF; line-height:1.3;
                                            margin-bottom:4px; overflow:hidden;
                                            display:-webkit-box; -webkit-line-clamp:2;
                                            -webkit-box-orient:vertical;">
                                    {name}
                                </div>
                                <div style="font-size:0.72rem; color:#8A7F74; margin-bottom:4px;">
                                    {gender} · {category}
                                </div>
                                <span class="badge badge-type">{article}</span>
                                <span class="badge badge-color">{colour}</span>
                                <div class="score-bar-container">
                                    <div class="score-label">Match: {score_pct}%</div>
                                    <div style="background:#E0DBD4; border-radius:2px; height:3px;">
                                        <div class="score-bar" style="width:{score_pct}%;"></div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Results table (expandable)
            with st.expander("View results as table"):
                display_cols = [
                    "id", "productDisplayName", "articleType",
                    "baseColour", "gender", "masterCategory",
                    "visual_score", "final_score"
                ]
                available = [c for c in display_cols if c in results_df.columns]
                st.dataframe(
                    results_df[available].style.format({"visual_score": "{:.3f}", "final_score": "{:.3f}"}),
                    use_container_width=True,
                )
else:
    # Empty state illustration
    st.markdown(
        """
        <div style="text-align:center; padding: 4rem 2rem; color:#9C8E80;">
            <div style="font-size:4rem; margin-bottom:1rem;"></div>
            <div style="font-family:'Playfair Display',serif; font-size:1.4rem;
                        color:#6B6357; margin-bottom:0.5rem;">
                Déposez une image de produit ci-dessus pour commencer.
            </div>
            <div style="font-size:0.9rem;">
                Trouvez les 10 produits les plus similaires visuellement dans votre catalogue.<br>
                dans votre catalogue grâce aux caractéristiques extraites par un CNN profond.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
