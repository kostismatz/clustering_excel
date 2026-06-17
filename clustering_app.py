import sys
import numpy as np
import pandas as pd
from collections import Counter
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import MiniBatchKMeans

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

    model_cluster = MiniBatchKMeans(
        n_clusters=20,
        batch_size=512,
        random_state=42
    )

    labels = model_cluster.fit_predict(X)

df["Cluster_ID"] = labels

# ===== Cluster Summary =====

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

    words = re.findall(r"[A-Za-zΑ-Ωα-ω]{3,}", texts.upper())

    stopwords = {
        "THE","AND","FOR","WITH","SET","PACK","DESIGN",
        "SMALL","LARGE","OF","TO","IN","ON",
        "ΚΑΙ","ΤΟ","ΤΗΝ","ΤΗΣ","ΓΙΑ","ΜΕ"
    }

    words = [w for w in words if w not in stopwords]

    top_words = [w for w, c in Counter(words).most_common(5)]

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

output = "clustered_output.xlsx"

with pd.ExcelWriter(output, engine="openpyxl") as writer:

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
