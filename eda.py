"""
eda.py
======
Exploratory Data Analysis. Every function returns a matplotlib Figure so it
can either be shown with plt.show() when run standalone, or passed straight
into st.pyplot() from app.py. All charts use seaborn styling, titles, axis
labels and legends as required.
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="viridis")


def class_distribution_bar(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = df["class_name"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index,
                legend=False, palette="viridis")
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Clothing Category")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def class_distribution_pie(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 7))
    counts = df["class_name"].value_counts()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=sns.color_palette("viridis", len(counts)))
    ax.set_title("Class Share of Dataset", fontsize=14, fontweight="bold")
    ax.legend(counts.index, title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig


def image_size_histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["width"], bins=20, alpha=0.6, label="Width")
    ax.hist(df["height"], bins=20, alpha=0.6, label="Height")
    ax.set_title("Image Size Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Pixels")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    return fig


def class_count_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(data=df, x="class_name", hue="class_name", legend=False,
                   order=df["class_name"].value_counts().index, ax=ax, palette="magma")
    ax.set_title("Count Plot per Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Clothing Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def numeric_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    numeric_cols = df[["width", "height", "file_size_kb"]]
    corr = numeric_cols.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, cbar=True, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix (Image Metadata)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def color_distribution_plot(df: pd.DataFrame, sample_n: int = 150):
    """Samples images and plots the average R/G/B channel distributions."""
    sample = df.sample(min(sample_n, len(df)), random_state=42)
    r_means, g_means, b_means = [], [], []
    for fpath in sample["filepath"]:
        img = cv2.imread(fpath)
        if img is None:
            continue
        b, g, r = cv2.mean(img)[:3]
        r_means.append(r); g_means.append(g); b_means.append(b)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(r_means, color="red", label="Red channel", ax=ax, fill=True, alpha=0.3)
    sns.kdeplot(g_means, color="green", label="Green channel", ax=ax, fill=True, alpha=0.3)
    sns.kdeplot(b_means, color="blue", label="Blue channel", ax=ax, fill=True, alpha=0.3)
    ax.set_title("Dominant Color Channel Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean Channel Intensity (0-255)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig


def most_common_category_summary(df: pd.DataFrame) -> dict:
    counts = df["class_name"].value_counts()
    return {
        "most_common_category": counts.idxmax(),
        "most_common_count": int(counts.max()),
        "least_common_category": counts.idxmin(),
        "least_common_count": int(counts.min()),
    }


def dataset_statistics_table(df: pd.DataFrame) -> pd.DataFrame:
    return df[["width", "height", "file_size_kb"]].describe().round(2)
