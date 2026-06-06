# =========================
# 1. IMPORT LIBRARIES
# =========================
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 2. APP CONFIG
# =========================
st.set_page_config(page_title="Learners Clustering App", layout="wide")

st.title("📊 Learners Clustering System")
st.write("Upload a dataset to cluster learners based on performance and behavior patterns.")

# =========================
# 3. FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # =========================
    # 4. DATA CLEANING
    # =========================

    if 'preferred_learning_style' in df.columns:
        df['preferred_learning_style'] = df['preferred_learning_style'].fillna(df['preferred_learning_style'].mode()[0])

    if 'english_level' in df.columns:
        df['english_level'] = df['english_level'].fillna(df['english_level'].mode()[0])

    if 'math_comfort_score_1_10' in df.columns:
        df["math_comfort_score_1_10"] = df["math_comfort_score_1_10"].fillna(df["math_comfort_score_1_10"].mean())

    if 'weekly_self_study_hours' in df.columns:
        df["weekly_self_study_hours"] = df["weekly_self_study_hours"].fillna(df["weekly_self_study_hours"].mean())

    if 'fee_balance_kes' in df.columns:
        df["fee_balance_kes"] = df["fee_balance_kes"].fillna(df["fee_balance_kes"].sum())

    if 'job_placement_readiness_score' in df.columns:
        df["job_placement_readiness_score"] = df["job_placement_readiness_score"].fillna(df["job_placement_readiness_score"].mean())

    # Drop unwanted columns if they exist
    for col in ["scholarship_status", "learner_id", "application_date"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # =========================
    # 5. ENCODING
    # =========================
    cat_cols = df.select_dtypes(include='object').columns

    encoder = LabelEncoder()

    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    # =========================
    # 6. SCALING
    # =========================
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

    # =========================
    # 7. K-MEANS CLUSTERING
    # =========================
    k = st.slider("Select number of clusters (K)", 2, 10, 5)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)

    df["cluster"] = clusters

    # =========================
    # 8. CLUSTER SUMMARY
    # =========================
    st.subheader("Cluster Summary (Mean Values)")
    cluster_summary = df.groupby("cluster").mean(numeric_only=True)
    st.dataframe(cluster_summary)

    # =========================
    # 9. PCA VISUALIZATION
    # =========================
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)

    pca_df = pd.DataFrame(pca_components, columns=["PC1", "PC2"])
    pca_df["cluster"] = df["cluster"]

    st.subheader("PCA Cluster Visualization")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="Set2",
        ax=ax
    )

    ax.set_title("K-Means Clusters (PCA Reduced Space)")
    st.pyplot(fig)

    # =========================
    # 10. CLUSTER LABELS
    # =========================
    cluster_labels = {
        0: "High Potential Achievers",
        1: "At-Risk Learners",
        2: "High Achievers / Job-Ready Professionals",
        3: "Consistent Mid Performers",
        4: "Emerging Learners / Developing Talent"
    }

    df["cluster_label"] = df["cluster"].map(cluster_labels)
    pca_df["cluster_label"] = pca_df["cluster"].map(cluster_labels)

    # =========================
    # 11. DOWNLOAD RESULTS
    # =========================
    st.subheader("Download Clustered Data")

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="clustered_learners.csv",
        mime="text/csv"
    )

    # =========================
    # 12. SHOW FINAL DATA
    # =========================
    st.subheader("Clustered Dataset Preview")
    st.dataframe(df.head())

else:
    st.info("Upload a CSV file to begin clustering.")