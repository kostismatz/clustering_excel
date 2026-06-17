import sys
import numpy as np
import pandas as pd
from collections import Counter
import re
import os

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import MiniBatchKMeans

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)


def run(file_path):

    df = pd.read_excel(file_path)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    text_cols = [c for c in df.columns if c not in numeric_cols]

    print(f"Text columns: {text_cols}")
    print(f"Numeric columns: {numeric_cols}")

    feature_blocks = []

    # ==========================
    # Numeric Features
    # ==========================

    if numeric_cols:

        num_df = df[numeric_cols].copy()

        num_df = num_df.fillna(num_df.median())

        scaler = StandardScaler()

        numeric_matrix = scaler.fit_transform(num_df)

        feature_blocks.append(numeric_matrix)

    # ==========================
    # Text Embeddings
    # ==========================

    for col in text_cols:

        print(f"Embedding column: {col}")

        texts = (
            df[col]
            .fillna("")
            .astype(str)
            .tolist()
        )

        emb = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True
        )

        feature_blocks.append(emb)

    # ==========================
    # Merge Features
    # ==========================

    X = np.hstack(feature_blocks)

    X = normalize(X)

    print("Feature matrix shape:", X.shape)

    # ==========================
    # Clustering
    # ==========================

    model_cluster = MiniBatchKMeans(
        n_clusters=20,
        batch_size=512,
        random_state=42
    )

    labels = model_cluster.fit_predict(X)

    df["Cluster_ID"] = labels

    # ==========================
    # Cluster Summary
    # ==========================

    cluster_summary = []

    for cluster_id in sorted(df["Cluster_ID"].unique()):

        cluster_df = df[df["Cluster_ID"] == cluster_id]

        texts = " ".join(

            cluster_df[text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()

        )

        words = re.findall(
            r"[A-Za-zΑ-Ωα-ω]{3,}",
            texts.upper()
        )

        stopwords = {

            "THE","AND","FOR","WITH",
            "SET","PACK","DESIGN",
            "SMALL","LARGE",
            "OF","TO","IN","ON",

            "ΚΑΙ","ΤΟ","ΤΗΝ",
            "ΤΗΣ","ΓΙΑ","ΜΕ",
            "ΑΠΟ","ΣΤΟ","ΣΤΗ"

        }

        words = [
            w for w in words
            if w not in stopwords
        ]

        top_words = [
            w for w, c
            in Counter(words).most_common(5)
        ]

        cluster_name = " / ".join(top_words[:3])

        business_description = (
            f"Products related to: {', '.join(top_words)}"
        )

        cluster_summary.append({

            "Cluster_ID": cluster_id,
            "Cluster_Name": cluster_name,
            "Business_Description": business_description,
            "Items": len(cluster_df)

        })

    summary_df = pd.DataFrame(cluster_summary)

    # ==========================
    # Save Excel
    # ==========================

    output = "clustered_output.xlsx"

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Clustered_Data",
            index=False
        )

        summary_df.to_excel(
            writer,
            sheet_name="Cluster_Summary",
            index=False
        )

    print("Saved:", output)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: clustering_app.exe input.xlsx"
        )
        sys.exit(1)

    run(sys.argv[1])
