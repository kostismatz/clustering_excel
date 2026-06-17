import sys
import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)


def run(file_path):

    df = pd.read_excel(file_path)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    text_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_matrix = None

    if numeric_cols:
        num_df = df[numeric_cols].fillna(df[numeric_cols].median())
        scaler = StandardScaler()
        numeric_matrix = scaler.fit_transform(num_df)

    text_data = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    embeddings = model.encode(text_data.tolist(), batch_size=64)

    if numeric_matrix is not None:
        X = np.hstack([numeric_matrix, embeddings])
    else:
        X = embeddings

    X = normalize(X)

   model_cluster = KMeans(
    n_clusters=10,
    random_state=42,
    n_init=10
)

labels = model_cluster.fit_predict(X)

    df["Cluster_ID"] = labels

    output = "clustered_output.xlsx"

    df.to_excel(output, index=False)

    print("Saved:", output)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: clustering_app.exe input.xlsx")
        sys.exit(1)

    run(sys.argv[1])
