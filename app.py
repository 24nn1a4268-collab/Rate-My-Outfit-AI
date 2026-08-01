"""
app.py
======
Rate My Outfit — AI Fashion Critic
Streamlit dashboard. Run with:  streamlit run app.py

Pages (sidebar navigation):
  1. Home
  2. Dataset Explorer      -> preprocessing.build_dataset_index / summary
  3. Preprocessing         -> preprocessing.py pipeline preview
  4. EDA                   -> eda.py charts
  5. Model Training        -> train.py results (run offline, viewed here)
  6. Rate My Outfit        -> predict.py full prediction dashboard
  7. Before vs After       -> compares two outfit photos
  8. History               -> past predictions saved this session
"""

import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import processing
import eda
import predict
from utils import ORGANIZED_DATASET_DIR, stars_from_score, CLASS_NAMES

# --------------------------------------------------------------------------
# PAGE CONFIG + DARK THEME
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Rate My Outfit — AI Fashion Critic",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #f0f0f0; }
    .metric-card {
        background: #1c1f26; padding: 1rem; border-radius: 12px;
        border: 1px solid #2a2e37; text-align: center;
    }
    .verdict-box {
        background: linear-gradient(135deg, #1c1f26, #2a2e37);
        padding: 1.5rem; border-radius: 16px; border: 1px solid #3a3f4b;
    }
    h1, h2, h3 { color: #f5c518 !important; }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------

st.sidebar.title("👗 Rate My Outfit")
st.sidebar.caption("AI Fashion Critic")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Abstract","Dataset Explorer", "Preprocessing", "EDA",
     "Model Training", "Rate My Outfit", "Before vs After"],
)

# --------------------------------------------------------------------------
# HOME
# --------------------------------------------------------------------------

if page == "Home":
    st.title("👗 Rate My Outfit — AI Fashion Critic")
    st.write(
        "An end-to-end AI + Computer Vision pipeline that classifies "
        "clothing, analyzes color and pattern, and uses a Vision-Language "
        "Model to roast or hype your outfit — with a full styling dashboard."
    )
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Pipeline Stages", "Clothing Classes", "Dashboard Sections", "VLM Backend"],
        ["5", str(len(CLASS_NAMES)), "19", "HF Transformers"],
    ):
        with col:
            st.markdown(f'<div class="metric-card"><h2>{value}</h2>{label}</div>',
                         unsafe_allow_html=True)
    st.info("Use the sidebar to explore the dataset, preprocessing, EDA, "
            "training results, or jump straight to **Rate My Outfit**.")

# --------------------------------------------------------------------------
# ABSTRACT 
# --------------------------------------------------------------------------

elif page == "Abstract":

    st.title("📄 Project Abstract")

    st.markdown("""
    ## 👗 Rate My Outfit – AI Fashion Critic

    ### 📌 Project Title
    **Rate My Outfit – AI-Based Clothing Detection and Outfit Rating System**

    ---

    ### 🎯 Objective

    The objective of this project is to build an Artificial Intelligence based
    fashion analysis system that can automatically detect clothing items from an
    uploaded image and provide an outfit rating based on the detected apparel.

    The system helps users evaluate their outfits instantly using Deep Learning
    and Computer Vision techniques.

    ---

    ### 🚀 Project Workflow

    **Step 1**
    - Collect clothing dataset

    **Step 2**
    - Organize images into class folders

    **Step 3**
    - Preprocess dataset
        - Resize images
        - Normalize pixel values
        - Remove duplicate images
        - Encode labels
        - Split into Train / Validation / Test

    **Step 4**
    - Train CNN Model

    **Step 5**
    - Save trained model (.h5)

    **Step 6**
    - Upload new clothing image

    **Step 7**
    - Predict clothing category

    **Step 8**
    - Display confidence score

    **Step 9**
    - Generate outfit rating and fashion feedback

    ---

    ### 🧠 Technologies Used

    - Python
    - Streamlit
    - TensorFlow
    - Keras
    - OpenCV
    - NumPy
    - Pandas
    - Scikit-Learn
    - Matplotlib
    - Pillow

    ---

    ### 📂 Dataset

    The dataset consists of different categories of clothing images such as:

    - 👕 T-Shirts
    - 👔 Shirts
    - 👖 Jeans
    - 👗 Dresses
    - 👟 Shoes
    - 🧥 Jackets
    - 🩳 Shorts
    - 👚 Skirts
    - 🧢 Caps
    - 🧶 Sweaters
    - 👖 Pants
    - 🎒 Accessories

    ---

    ### ⚙️ Preprocessing Techniques

    ✔ Image Resizing

    ✔ Normalization

    ✔ Duplicate Image Removal

    ✔ Label Encoding

    ✔ One-Hot Encoding

    ✔ Data Augmentation

    ✔ Train-Test Split

    ---

    ### 🤖 Model Used

    Convolutional Neural Network (CNN)

    Model Layers:

    • Convolution Layer

    • MaxPooling Layer

    • Dropout Layer

    • Dense Layer

    • Softmax Output Layer

    ---

    ### ✨ Features

    ✔ Clothing Detection

    ✔ Confidence Score

    ✔ Outfit Rating

    ✔ Fashion Feedback

    ✔ Dataset Explorer

    ✔ Model Training Dashboard

    ✔ Before vs After Prediction

    ✔ Attractive Streamlit Interface

    ---

    ### 📊 Output

    The application predicts:

    • Clothing Category

    • Prediction Confidence

    • Outfit Rating (/10)

    • Fashion Feedback

    ---

    ### 🌟 Future Enhancements

    • Detect multiple clothing items

    • Color Combination Analysis

    • Fashion Recommendation System

    • Outfit Matching Suggestions

    • Celebrity Style Comparison

    • Seasonal Outfit Recommendation

    • Mobile Application Deployment

    ---

    ### 👨‍💻 Conclusion

    Rate My Outfit demonstrates how Artificial Intelligence and Computer Vision
    can simplify fashion analysis by automatically recognizing clothing items
    and providing instant outfit ratings. The project integrates data
    preprocessing, deep learning, model prediction, and an interactive
    Streamlit interface into one complete AI application.
    """)

# --------------------------------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------------------------------

elif page == "Dataset Explorer":
    st.title("📂 Dataset Module")
    if not os.path.isdir(ORGANIZED_DATASET_DIR) or not os.listdir(ORGANIZED_DATASET_DIR):
        st.warning(
            f"No organized dataset found at `{ORGANIZED_DATASET_DIR}/`. "
            "Run `python organize_dataset.py` first to auto-sort your raw "
            "images into class folders."
        )
    else:
        with st.spinner("Indexing dataset..."):
            df = processing.build_dataset_index()
            summary = processing.dataset_summary(df)

        if summary.get("is_empty"):
            st.error(
                f"`{ORGANIZED_DATASET_DIR}/` exists but no readable images were "
                "found inside its class subfolders. This usually means:\n\n"
                "- `organize_dataset.py` ran but skipped every image (try lowering "
                "`--min-confidence`), or\n"
                "- the class folders are empty / images are in an unsupported "
                "format, or\n"
                "- the app is running from a different working directory than "
                "the `dataset/` folder.\n\n"
                "Check the terminal running `streamlit run app.py` — "
                "`build_dataset_index` prints exactly what it skipped and why."
            )
            st.stop()

        c1, c2, c3 = st.columns(3)
        c1.metric("Dataset Shape", f"{summary['shape'][0]} × {summary['shape'][1]}")
        c2.metric("Number of Images", summary["num_images"])
        c3.metric("Number of Classes", summary["num_classes"])

        st.subheader("Class Names")
        st.write(", ".join(summary["class_names"]))

        st.subheader("Dataset Info")
        st.json({"dtypes": summary["dtypes"]})

        c1, c2 = st.columns(2)
        c1.metric("Missing Values", sum(summary["missing_values"].values()))
        c2.metric("Duplicate Images", summary["duplicate_rows"])

        st.subheader("First Five Rows")
        st.dataframe(summary["head"])

        st.subheader("Random Sample Images")
        sample = df.sample(min(8, len(df)))
        cols = st.columns(4)
        for i, (_, row) in enumerate(sample.iterrows()):
            with cols[i % 4]:
                st.image(row["filepath"], caption=row["class_name"], use_container_width=True)

# --------------------------------------------------------------------------
# PREPROCESSING
# --------------------------------------------------------------------------

elif page == "Preprocessing":
    st.title("🧹 Preprocessing Module")
    if not os.path.isdir(ORGANIZED_DATASET_DIR) or not os.listdir(ORGANIZED_DATASET_DIR):
        st.warning("Organize your dataset first (see Dataset Explorer page).")
    else:
        df = processing.build_dataset_index()
        if df.empty or "class_name" not in df.columns:
            st.error(
                f"No readable images found under `{ORGANIZED_DATASET_DIR}/`. "
                "Visit the Dataset Explorer page first — it explains the "
                "likely cause."
            )
            st.stop()

        st.write(f"Loaded index of **{len(df)}** images across "
                 f"**{df['class_name'].nunique()}** classes.")

        if st.button("Run Preprocessing Preview"):
            with st.spinner("Removing duplicates, resizing, normalizing, encoding..."):
                df_clean = processing.remove_duplicate_images(df)
                sample_df = df_clean
                X, y_int, encoder = processing.load_images_and_labels(sample_df)
                y_cat = processing.encode_labels_categorical(y_int, len(encoder.classes_))
                X_train, X_val, X_test, y_train, y_val, y_test = processing.split_dataset(X, y_cat)

            c1, c2, c3 = st.columns(3)
            c1.metric("Training Images", len(X_train))
            c2.metric("Validation Images", len(X_val))
            c3.metric("Testing Images", len(X_test))

            st.subheader("First Five Processed Images")
            cols = st.columns(5)
            for i in range(min(5, len(X_train))):
                with cols[i]:
                    st.image(X_train[i], caption="Resized + Normalized", use_container_width=True)

            st.success("Preprocessing complete: resized to 224×224, normalized to [0,1], "
                       "labels one-hot encoded, duplicates removed, train/val/test split done.")

# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------

elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    if not os.path.isdir(ORGANIZED_DATASET_DIR) or not os.listdir(ORGANIZED_DATASET_DIR):
        st.warning("Organize your dataset first (see Dataset Explorer page).")
    else:
        df = processing.build_dataset_index()
        if df.empty or "class_name" not in df.columns:
            st.error(
                f"No readable images found under `{ORGANIZED_DATASET_DIR}/`. "
                "Visit the Dataset Explorer page first — it explains the "
                "likely cause."
            )
            st.stop()

        st.subheader("Class Distribution")
        st.pyplot(eda.class_distribution_bar(df))
        st.pyplot(eda.class_distribution_pie(df))

        st.subheader("Count Plot")
        st.pyplot(eda.class_count_plot(df))

        st.subheader("Image Size Distribution")
        st.pyplot(eda.image_size_histogram(df))

        st.subheader("Color Distribution")
        st.pyplot(eda.color_distribution_plot(df))

        st.subheader("Correlation Matrix")
        st.pyplot(eda.numeric_correlation_heatmap(df))

        st.subheader("Most Common Category")
        st.json(eda.most_common_category_summary(df))

        st.subheader("Dataset Statistics")
        st.dataframe(eda.dataset_statistics_table(df))

# --------------------------------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------------------------------

elif page == "Model Training":
    st.title("🧠 Model Training & Evaluation")
    st.write(
        "Training is compute-heavy (transfer learning over the full dataset) "
        "so it's meant to be run once from the command line:"
    )
    st.code("python train.py", language="bash")
    st.write(
        "`train.py` builds a MobileNetV2 (default) / EfficientNetB0 / ResNet50 "
        "transfer-learning classifier, trains it, evaluates accuracy, "
        "precision, recall, F1, confusion matrix, classification report and "
        "ROC/AUC, then saves the model to `saved_models/outfit_classifier.h5`. "
        "Come back to this page after training to review metrics saved as "
        "`reports/metrics.json` (see README.md for the exact CLI walkthrough)."
    )

    metrics_path = "reports/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{metrics['precision']*100:.1f}%")
        c3.metric("Recall", f"{metrics['recall']*100:.1f}%")
        c4.metric("F1 Score", f"{metrics['f1_score']*100:.1f}%")
    else:
        st.info("No saved metrics yet. Run train.py to populate this page.")

# --------------------------------------------------------------------------
# RATE MY OUTFIT (main feature)
# --------------------------------------------------------------------------

elif page == "Rate My Outfit":
    st.title("🪞 Rate My Outfit")
    st.write("Upload a mirror selfie or take a photo — the AI critic will handle the rest.")

    tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Camera Capture"])
    image_file = None
    with tab1:
        image_file = st.file_uploader("Upload your outfit photo", type=["jpg", "jpeg", "png"])
    with tab2:
        cam_file = st.camera_input("Take a photo")
        if cam_file is not None:
            image_file = cam_file

    if image_file is not None:
        pil_image = Image.open(image_file)
        st.image(pil_image, caption="Uploaded Image", use_container_width=False, width=350)

        if st.button("✨ Rate This Outfit", type="primary"):
            try:
                with st.spinner("Analyzing outfit..."):
                    result = predict.run_full_prediction(pil_image)
            except FileNotFoundError as e:
                st.error(str(e))
                st.stop()

            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "overall_score": result["overall_score"],
                "items": [d["item"] for d in result["detections"]],
            })

            # 3 & 4. Roast / Hype
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔥 AI Roast Mode")
                for line in result["roast"]:
                    st.write(f"> {line}")
            with col2:
                st.subheader("💫 Hype Mode")
                for line in result["hype"]:
                    st.write(f"> {line}")

            # 5. Outfit Rating
            st.subheader("⭐ Outfit Rating")
            rc1, rc2 = st.columns(2)
            items = list(result["ratings"].items())
            for i, (label, score) in enumerate(items):
                target = rc1 if i % 2 == 0 else rc2
                with target:
                    st.write(f"**{label}**: {score}/10  {stars_from_score(score)}")
                    st.progress(score / 10)


            # 7. Color Combination Analysis
            st.subheader("🎨 Color Combination Analysis")
            cc1, cc2 = st.columns([1, 2])
            with cc1:
                for c in result["colors"]:
                    st.color_picker(f"{c['name'].title()} ({c['pct']}%)",
                                     '#%02x%02x%02x' % c["rgb"], disabled=True)
            with cc2:
                st.write(f"**Match Rating:** {result['color_rating']}")
                st.write(result["color_suggestion"])
                st.write(f"**Detected Pattern:** {result['pattern']}")

            # 8. Occasion Recommendation
            st.subheader("🎉 Occasion Recommendation")
            st.write(" • ".join(result["occasions"]))

            # 9. Fashion Tips
            st.subheader("💡 Fashion Tips")
            for tip in result["fashion_tips"]:
                st.write(f"- {tip}")

            # 10. Missing Accessories
            st.subheader("🕶️ Missing Accessories")
            st.write(" • ".join(result["missing_accessories"]))

            # 11. Trend Score
            st.subheader("📈 Trend Score")
            tcols = st.columns(len(result["trend_scores"]))
            for col, (label, score) in zip(tcols, result["trend_scores"].items()):
                col.metric(label, f"{score}/10")

            # 12. Mood Detection
            st.subheader("🎭 Outfit Mood Detection")
            st.write(" • ".join(result["mood"]))

            # 13. Weather Compatibility
            st.subheader("🌦️ Weather Compatibility")
            st.write(f"Best suited for: **{result['weather']}**")

            # 14. Fabric Suggestion
            st.subheader("🧵 Fabric Suggestion")
            st.write(f"Likely fabric: **{result['fabric']}**")

            # 15. Confidence Meter
            st.subheader("🎯 Confidence Meter")
            top_conf = result["detections"][0]["confidence"] if result["detections"] else 0
            st.progress(top_conf, text=f"{top_conf*100:.1f}% classifier confidence")

            # 16. Fashion Dashboard (radar-style via bar chart, score cards)
            st.subheader("📊 Fashion Dashboard")
            dash_df = pd.DataFrame(
                {"Score": list(result["ratings"].values())},
                index=list(result["ratings"].keys()),
            )
            st.bar_chart(dash_df)

            # 19. Final Verdict
            st.markdown("---")
            st.markdown('<div class="verdict-box">', unsafe_allow_html=True)
            st.subheader("🏆 Final Verdict")
            st.write(stars_from_score(result["overall_score"]))
            st.write(f"**Overall Rating: {result['overall_score']}/10**")


# --------------------------------------------------------------------------
# BEFORE VS AFTER
# --------------------------------------------------------------------------

elif page == "Before vs After":
    st.title("🔁 Before vs After Comparison")
    c1, c2 = st.columns(2)
    with c1:
        img_a = st.file_uploader("Outfit A", type=["jpg", "jpeg", "png"], key="a")
    with c2:
        img_b = st.file_uploader("Outfit B", type=["jpg", "jpeg", "png"], key="b")

    if img_a and img_b and st.button("Compare Outfits"):
        try:
            with st.spinner("Analyzing both outfits..."):
                result_a = predict.run_full_prediction(Image.open(img_a))
                result_b = predict.run_full_prediction(Image.open(img_b))
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        c1, c2 = st.columns(2)
        with c1:
            st.image(img_a, caption=f"Outfit A — {result_a['overall_score']}/10", use_container_width=True)
        with c2:
            st.image(img_b, caption=f"Outfit B — {result_b['overall_score']}/10", use_container_width=True)

        winner = "Outfit A" if result_a["overall_score"] >= result_b["overall_score"] else "Outfit B"
        improvement = round(abs(result_a["overall_score"] - result_b["overall_score"]), 1)

        st.subheader("🏆 Result")
        st.write(f"**Winner:** {winner}")
        st.write(f"**Score Gap:** {improvement} points")
        st.write(f"**Reason:** {winner} scores higher on overall styling, color "
                 f"harmony and presentation based on the detected items and palette.")
